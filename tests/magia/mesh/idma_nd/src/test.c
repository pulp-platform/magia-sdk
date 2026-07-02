// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Functional test for idma_memcpy_md_to_nd, the runtime-dispatched md->nd IDMA
// data mover. Source and destination are described by independent tensor
// sub-slices (possibly different rank/shape, same element count); the mover
// prefers native 2D bursts (fast path, when the local-L1/OBI side is packed) and
// falls back to 1D bursts otherwise.
//
// Only tile 0 exercises the mover (it is a per-tile library routine); every tile
// then joins a global barrier so the mesh terminates cleanly. For each case the
// source region is filled with a per-element byte pattern, the copy is run, and
// the whole destination extent is compared byte-for-byte against a CPU-computed
// reference image. Because untouched destination bytes are pre-set to a sentinel
// and the reference keeps that sentinel outside written elements, the comparison
// catches under-copy, over-copy, and wrong-address writes alike.

#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"

#define WAIT_MODE WFE
#define SENTINEL  0xEEu

// L2-resident scratch (uninitialized globals land in dataram = L2, reached via
// AXI). One buffer plays the AXI side of a copy; the other holds the reference.
static uint8_t l2_axi[IDMA_ND_BUF_BYTES];
static uint8_t l2_ref[IDMA_ND_BUF_BYTES];

static inline void poke8(uint32_t addr, uint8_t v) { *(volatile uint8_t *)addr = v; }

// Kept out-of-line on purpose: at -O2 this PULP GCC miscompiles the volatile
// byte-load when it is inlined straight into a comparison (the loaded byte is
// left sign-extended), so folding peek8 into `!=` yields spurious mismatches.
static __attribute__((noinline)) uint32_t peek8(uint32_t addr)
{
    return (uint32_t)(*(volatile uint8_t *)addr) & 0xFFu;
}

static void fill_bytes(uint32_t base, uint8_t val, uint32_t n)
{
    for (uint32_t i = 0; i < n; ++i) {
        poke8(base + i, val);
    }
}

// Deterministic, per-(element, byte) pattern; distinct across elements.
static inline uint8_t pattern_byte(uint32_t k, uint32_t j, uint32_t elem_bytes)
{
    return (uint8_t)(0x11u + k * elem_bytes + j);
}

// Byte offset of row-major linear element k for a descriptor, honoring per-dim
// starts (mirrors the addressing the mover performs).
static uint32_t desc_off(const tensor_sub_slice_t *d, uint32_t k)
{
    uint32_t off = 0u;
    for (uint32_t dd = d->rank; dd-- > 0u;) {
        uint32_t idx = k % d->dims[dd].length;
        k /= d->dims[dd].length;
        off += (d->dims[dd].start + idx) * d->dims[dd].stride;
    }
    return off;
}

// Run one copy case and return the number of mismatched destination bytes.
static uint32_t run_case(idma_controller_t *idma,
                         eu_controller_t *eu,
                         uint8_t dir,
                         const tensor_sub_slice_t *src,
                         const tensor_sub_slice_t *dst,
                         uint32_t elem_bytes,
                         uint32_t l1_base)
{
    uint32_t axi      = (uint32_t)l2_axi;
    uint32_t obi      = l1_base;
    uint32_t src_base = dir ? obi : axi; // dir=1: OBI is source; dir=0: AXI is source
    uint32_t dst_base = dir ? axi : obi;
    uint32_t n        = src->num_elems;

    // Sentinel-fill source (so strided gaps are defined), destination, and ref.
    fill_bytes(src_base, SENTINEL, IDMA_ND_BUF_BYTES);
    fill_bytes(dst_base, SENTINEL, IDMA_ND_BUF_BYTES);
    for (uint32_t i = 0; i < IDMA_ND_BUF_BYTES; ++i) {
        l2_ref[i] = SENTINEL;
    }

    // Write the pattern into source element slots and into the reference image
    // at the destination element slots.
    for (uint32_t k = 0; k < n; ++k) {
        uint32_t so = desc_off(src, k);
        uint32_t di = desc_off(dst, k);
        for (uint32_t j = 0; j < elem_bytes; ++j) {
            poke8(src_base + so + j, pattern_byte(k, j, elem_bytes));
            l2_ref[di + j] = pattern_byte(k, j, elem_bytes);
        }
    }

    idma_memcpy_md_to_nd(idma, dir, dst_base, src_base, src, dst, elem_bytes, eu);

    // Materialize both operands into locals before comparing: folding the
    // volatile byte-load directly into the `!=` miscompiles at -O2 on the PULP
    // GCC used here (the loaded byte is left sign-extended).
    uint32_t errs = 0u;
    for (uint32_t i = 0; i < IDMA_ND_BUF_BYTES; ++i) {
        uint32_t dv = peek8(dst_base + i);
        uint32_t rv = (uint32_t)l2_ref[i];
        if (dv != rv) {
            errs++;
        }
    }
    return errs;
}

int main(void)
{
    uint32_t hartid = get_hartid();

    idma_config_t idma_cfg   = {.hartid = hartid};
    idma_controller_t idma_ctrl = {.base = NULL, .cfg = &idma_cfg, .api = &idma_api};
    idma_init(&idma_ctrl);

    fsync_config_t fsync_cfg     = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {.base = NULL, .cfg = &fsync_cfg, .api = &fsync_api};
    fsync_init(&fsync_ctrl);

#if STALLING == 0
    eu_config_t eu_cfg       = {.hartid = hartid};
    eu_controller_t eu_ctrl  = {.base = NULL, .cfg = &eu_cfg, .api = &eu_api};
    eu_init(&eu_ctrl);
    eu_clear_events(0xFFFFFFFF);
    eu_fsync_init(&eu_ctrl, 0);
    eu_idma_init(&eu_ctrl, 0);
    eu_controller_t *eu = &eu_ctrl;
#else
    eu_controller_t *eu = NULL;
#endif

    uint32_t errors = 0u;

    if (hartid == 0u) {
        uint32_t l1 = get_l1_base(hartid);

        uint32_t case_errs;

        // --- C1: fully-contiguous 1D -> 1D, both directions (native 1D) --------
        printf("=== C1: fully-contiguous 1D <-> 1D, both directions ===\n");
        case_errs = errors;
        {
            tensor_sub_slice_t a = {.rank = 1u, .num_elems = 16u, .dims = {{0u, 16u, 2u}}};
            errors += run_case(&idma_ctrl, eu, 0u, &a, &a, 2u, l1);
            errors += run_case(&idma_ctrl, eu, 1u, &a, &a, 2u, l1);
        }
        printf("[C1] errors: %d\n", errors - case_errs);

        // --- C2: 2D strided L2 <-> contiguous L1 (native 2D), both directions --
        printf("=== C2: 2D strided L2 <-> contiguous L1 (native 2D), both directions ===\n");
        case_errs = errors;
        {
            tensor_sub_slice_t strided = {
                .rank = 2u, .num_elems = 24u, .dims = {{0u, 6u, 16u}, {0u, 4u, 2u}}};
            tensor_sub_slice_t packed = {
                .rank = 2u, .num_elems = 24u, .dims = {{0u, 6u, 8u}, {0u, 4u, 2u}}};
            // dir=0 (load): AXI=src strided, OBI=dst packed L1.
            errors += run_case(&idma_ctrl, eu, 0u, &strided, &packed, 2u, l1);
            // dir=1 (store): OBI=src packed L1, AXI=dst strided.
            errors += run_case(&idma_ctrl, eu, 1u, &packed, &strided, 2u, l1);
        }
        printf("[C2] errors: %d\n", errors - case_errs);

        // --- C3: reshape with different rank/shape, fully contiguous (-> 1D) ----
        printf("=== C3: reshape with different rank/shape, fully contiguous (-> 1D) ===\n");
        case_errs = errors;
        {
            tensor_sub_slice_t s1 = {.rank = 1u, .num_elems = 12u, .dims = {{0u, 12u, 2u}}};
            tensor_sub_slice_t d2 = {
                .rank = 2u, .num_elems = 12u, .dims = {{0u, 3u, 8u}, {0u, 4u, 2u}}};
            errors += run_case(&idma_ctrl, eu, 0u, &s1, &d2, 2u, l1);

            tensor_sub_slice_t s2 = {
                .rank = 2u, .num_elems = 12u, .dims = {{0u, 2u, 12u}, {0u, 6u, 2u}}};
            errors += run_case(&idma_ctrl, eu, 0u, &s2, &d2, 2u, l1);
        }
        printf("[C3] errors: %d\n", errors - case_errs);

        // --- C4: strided AXI side with rank>=2 outer dims (-> loop of 2D) ------
        printf("=== C4: strided AXI side with rank>=2 outer dims (-> loop of 2D) ===\n");
        case_errs = errors;
        {
            // src (AXI) coalesces to {outer(2,std48), mid(3,std12), inner(4,std2)}:
            // inner is the contiguous burst block, mid/outer are strided levels.
            tensor_sub_slice_t s3 = {
                .rank = 3u, .num_elems = 24u, .dims = {{0u, 2u, 48u}, {0u, 3u, 12u}, {0u, 4u, 2u}}};
            tensor_sub_slice_t dpk = {.rank = 1u, .num_elems = 24u, .dims = {{0u, 24u, 2u}}};
            errors += run_case(&idma_ctrl, eu, 0u, &s3, &dpk, 2u, l1);
        }
        printf("[C4] errors: %d\n", errors - case_errs);

        // --- C5: non-zero per-dim starts, contiguous ---------------------------
        printf("=== C5: non-zero per-dim starts, contiguous ===\n");
        case_errs = errors;
        {
            tensor_sub_slice_t s = {.rank = 1u, .num_elems = 4u, .dims = {{2u, 4u, 2u}}};
            tensor_sub_slice_t d = {.rank = 1u, .num_elems = 4u, .dims = {{1u, 4u, 2u}}};
            errors += run_case(&idma_ctrl, eu, 0u, &s, &d, 2u, l1);
        }
        printf("[C5] errors: %d\n", errors - case_errs);

        // --- C6: non-contiguous innermost on both sides (-> 1D odometer path) --
        printf("=== C6: non-contiguous innermost on both sides (-> 1D odometer path) ===\n");
        case_errs = errors;
        {
            // dir=0 so OBI=dst; its inner stride (4) != elem (2) -> not packed,
            // forcing the software fallback over the common contiguous run (1).
            tensor_sub_slice_t s = {
                .rank = 2u, .num_elems = 12u, .dims = {{0u, 3u, 20u}, {0u, 4u, 4u}}};
            tensor_sub_slice_t d = {
                .rank = 2u, .num_elems = 12u, .dims = {{0u, 4u, 16u}, {0u, 3u, 4u}}};
            errors += run_case(&idma_ctrl, eu, 0u, &s, &d, 2u, l1);
        }
        printf("[C6] errors: %d\n", errors - case_errs);

        // --- C7: max rank (IDMA_ND_MAX_RANK) accepted, contiguous -> 1D --------
        printf("=== C7: max rank (IDMA_ND_MAX_RANK) accepted, contiguous -> 1D ===\n");
        case_errs = errors;
        {
            tensor_sub_slice_t s4 = {.rank = 4u,
                                     .num_elems = 16u,
                                     .dims = {{0u, 2u, 16u}, {0u, 2u, 8u}, {0u, 2u, 4u}, {0u, 2u, 2u}}};
            tensor_sub_slice_t d1 = {.rank = 1u, .num_elems = 16u, .dims = {{0u, 16u, 2u}}};
            errors += run_case(&idma_ctrl, eu, 0u, &s4, &d1, 2u, l1);
        }
        printf("[C7] errors: %d\n", errors - case_errs);

        // --- C8: 4-byte elements, contiguous and 2D strided --------------------
        printf("=== C8: 4-byte elements, contiguous and 2D strided ===\n");
        case_errs = errors;
        {
            tensor_sub_slice_t a = {.rank = 1u, .num_elems = 8u, .dims = {{0u, 8u, 4u}}};
            errors += run_case(&idma_ctrl, eu, 0u, &a, &a, 4u, l1);

            tensor_sub_slice_t strided = {
                .rank = 2u, .num_elems = 12u, .dims = {{0u, 4u, 20u}, {0u, 3u, 4u}}};
            tensor_sub_slice_t packed = {
                .rank = 2u, .num_elems = 12u, .dims = {{0u, 4u, 12u}, {0u, 3u, 4u}}};
            errors += run_case(&idma_ctrl, eu, 0u, &strided, &packed, 4u, l1);
        }
        printf("[C8] errors: %d\n", errors - case_errs);

        // --- C9: empty transfer (a zero-length dim) is a no-op -----------------
        printf("=== C9: empty transfer (a zero-length dim) is a no-op ===\n");
        case_errs = errors;
        {
            uint32_t dst_base = l1;
            fill_bytes(dst_base, SENTINEL, IDMA_ND_BUF_BYTES);
            tensor_sub_slice_t s = {.rank = 1u, .num_elems = 0u, .dims = {{0u, 0u, 2u}}};
            tensor_sub_slice_t d = {.rank = 1u, .num_elems = 0u, .dims = {{0u, 0u, 2u}}};
            int ret = idma_memcpy_md_to_nd(&idma_ctrl, 0u, dst_base, (uint32_t)l2_axi, &s, &d, 2u, eu);
            if (ret != 0) {
                errors++;
            }
            for (uint32_t i = 0; i < IDMA_ND_BUF_BYTES; ++i) {
                uint32_t dv = peek8(dst_base + i);
                if (dv != (uint32_t)SENTINEL) {
                    errors++;
                }
            }
        }
        printf("[C9] errors: %d\n", errors - case_errs);

        // --- C10: mismatched element counts is rejected (-1), leaves dst intact -
        printf("=== C10: mismatched element counts is rejected (-1), leaves dst intact ===\n");
        case_errs = errors;
        {
            uint32_t dst_base = l1;
            fill_bytes(dst_base, SENTINEL, IDMA_ND_BUF_BYTES);
            tensor_sub_slice_t s = {.rank = 1u, .num_elems = 4u, .dims = {{0u, 4u, 2u}}};
            tensor_sub_slice_t d = {.rank = 1u, .num_elems = 8u, .dims = {{0u, 8u, 2u}}};
            int ret = idma_memcpy_md_to_nd(&idma_ctrl, 0u, dst_base, (uint32_t)l2_axi, &s, &d, 2u, eu);
            if (ret != -1) {
                errors++;
            }
            for (uint32_t i = 0; i < IDMA_ND_BUF_BYTES; ++i) {
                uint32_t dv = peek8(dst_base + i);
                if (dv != (uint32_t)SENTINEL) {
                    errors++;
                }
            }
        }
        printf("[C10] errors: %d\n", errors - case_errs);

        printf("test_idma_nd errors: %d\n", errors);
    }

    // Global barrier so every tile terminates together.
    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    return errors;
}
