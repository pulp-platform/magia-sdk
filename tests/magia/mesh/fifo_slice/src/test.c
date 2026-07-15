// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Functional test for the generalized L1 FIFO mailbox (utils/l1_fifo.h) carrying
// tensor_sub_slice_t descriptors. One producer tile pushes a series of sub-slice
// messages into a consumer tile's FIFO via fifo_push (which gathers a possibly
// strided local region into the consumer slot's packed payload through
// idma_memcpy_md_to_nd). The consumer pops each message, checks the round-tripped
// descriptor/metadata, and byte-checks the packed payload against a recomputed
// per-element reference. No golden file: the pattern is generated at runtime.
//
// Cases cover contiguous 1-D, packed 2-D, strided 2-D gather (software fallback),
// reshape (src rank != published rank), non-zero per-dim start, max rank, 4-byte
// elements, a zero-length no-op, and a producer-only negative case (rank beyond
// IDMA_ND_MAX_RANK must be rejected).
//
// Only tiles 1 (producer) and 0 (consumer) do work; every tile joins the
// barriers so the mesh terminates cleanly. Runs at any tile count >= 2.

#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"
#include "utils/l1_fifo.h"

#define WAIT_MODE WFE
#define SENTINEL  0xEEu

#define NUM_TILES (MESH_X_TILES * MESH_Y_TILES)

typedef struct {
    tensor_sub_slice_t src;  // producer-local gather layout (may be strided)
    tensor_sub_slice_t desc; // logical descriptor published to the consumer
    uint32_t elem_bytes;
    uint32_t src_off; // byte offset of the region base within the producer buffer
    const char *name;
} slice_case_t;

static const slice_case_t CASES[FIFO_SLICE_NUM_CASES] = {
    // C0: fully-contiguous 1-D, 16 x 2B.
    {.src   = {.rank = 1u, .num_elems = 16u, .dims = {{0u, 16u, 2u}}},
     .desc  = {.rank = 1u, .num_elems = 16u, .dims = {{0u, 16u, 2u}}},
     .elem_bytes = 2u,
     .src_off    = 0u,
     .name       = "contig-1d"},
    // C1: packed 2-D 4x8 (row stride == packed), 2B.
    {.src   = {.rank = 2u, .num_elems = 32u, .dims = {{0u, 4u, 16u}, {0u, 8u, 2u}}},
     .desc  = {.rank = 2u, .num_elems = 32u, .dims = {{0u, 4u, 16u}, {0u, 8u, 2u}}},
     .elem_bytes = 2u,
     .src_off    = 0u,
     .name       = "packed-2d"},
    // C2: strided 2-D source (row stride 32 > packed 12) -> software fallback.
    {.src   = {.rank = 2u, .num_elems = 24u, .dims = {{0u, 4u, 32u}, {0u, 6u, 2u}}},
     .desc  = {.rank = 2u, .num_elems = 24u, .dims = {{7u, 4u, 12u}, {0u, 6u, 2u}}},
     .elem_bytes = 2u,
     .src_off    = 0u,
     .name       = "strided-2d"},
    // C3: reshape, src rank 2 -> published rank 1, same 12 elems.
    {.src   = {.rank = 2u, .num_elems = 12u, .dims = {{0u, 2u, 24u}, {0u, 6u, 2u}}},
     .desc  = {.rank = 1u, .num_elems = 12u, .dims = {{0u, 12u, 2u}}},
     .elem_bytes = 2u,
     .src_off    = 0u,
     .name       = "reshape"},
    // C4: non-zero per-dim start on both sides.
    {.src   = {.rank = 1u, .num_elems = 4u, .dims = {{2u, 4u, 2u}}},
     .desc  = {.rank = 1u, .num_elems = 4u, .dims = {{1u, 4u, 2u}}},
     .elem_bytes = 2u,
     .src_off    = 0u,
     .name       = "nonzero-start"},
    // C5: max rank (IDMA_ND_MAX_RANK) source, published as 1-D.
    {.src   = {.rank      = 4u,
               .num_elems = 16u,
               .dims      = {{0u, 2u, 16u}, {0u, 2u, 8u}, {0u, 2u, 4u}, {0u, 2u, 2u}}},
     .desc  = {.rank = 1u, .num_elems = 16u, .dims = {{0u, 16u, 2u}}},
     .elem_bytes = 2u,
     .src_off    = 0u,
     .name       = "max-rank"},
    // C6: 4-byte elements, contiguous 1-D.
    {.src   = {.rank = 1u, .num_elems = 8u, .dims = {{0u, 8u, 4u}}},
     .desc  = {.rank = 1u, .num_elems = 8u, .dims = {{0u, 8u, 4u}}},
     .elem_bytes = 4u,
     .src_off    = 0u,
     .name       = "elem-4b"},
    // C7: zero-length transfer (a zero-length dim) is a no-op that still publishes.
    {.src   = {.rank = 1u, .num_elems = 0u, .dims = {{0u, 0u, 2u}}},
     .desc  = {.rank = 1u, .num_elems = 0u, .dims = {{0u, 0u, 2u}}},
     .elem_bytes = 2u,
     .src_off    = 0u,
     .name       = "zero-len"},
};

static inline void poke8(uint32_t addr, uint8_t v) { *(volatile uint8_t *)addr = v; }

// Kept out-of-line: at -O2 this PULP GCC miscompiles the inlined volatile byte
// load folded into a comparison (the loaded byte is left sign-extended).
static __attribute__((noinline)) uint32_t peek8(uint32_t addr)
{
    return (uint32_t)(*(volatile uint8_t *)addr) & 0xFFu;
}

static void fill_bytes(uint32_t base, uint8_t val, uint32_t n)
{
    for (uint32_t i = 0; i < n; ++i)
        poke8(base + i, val);
}

// Deterministic per-(case, element, byte) pattern; distinct across all three.
static inline uint8_t pattern_byte(uint32_t c, uint32_t k, uint32_t j)
{
    return (uint8_t)(0x11u + c * 37u + k * 7u + j);
}

// Byte offset of row-major linear element k for a descriptor, honoring per-dim
// starts (mirrors the addressing the mover performs; same as idma_nd).
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

static uint32_t desc_equal(const tensor_sub_slice_t *a, const tensor_sub_slice_t *b)
{
    if (a->rank != b->rank || a->num_elems != b->num_elems)
        return 0u;
    for (uint32_t d = 0; d < a->rank; ++d) {
        if (a->dims[d].start != b->dims[d].start || a->dims[d].length != b->dims[d].length ||
            a->dims[d].stride != b->dims[d].stride)
            return 0u;
    }
    return 1u;
}

// Consumer-side check of one received message against its case definition.
static uint32_t verify_case(uint32_t c, const fifo_msg_t *msg)
{
    const slice_case_t *cs = &CASES[c];
    uint32_t errs          = 0u;

    if (msg->elem_bytes != cs->elem_bytes)
        errs++;
    if (msg->data_size != cs->desc.num_elems * cs->elem_bytes)
        errs++;
    if (!desc_equal(&msg->desc, &cs->desc))
        errs++;

    // Packed payload: element k of the source (row-major) landed at byte
    // k * elem_bytes and must carry the source element's pattern.
    for (uint32_t k = 0; k < cs->desc.num_elems; ++k) {
        for (uint32_t j = 0; j < cs->elem_bytes; ++j) {
            uint32_t got = peek8(msg->data_ptr + k * cs->elem_bytes + j);
            uint32_t exp = pattern_byte(c, k, j);
            if (got != exp)
                errs++;
        }
    }
    return errs;
}

int main(void)
{
    uint32_t hartid = get_hartid();

    idma_config_t idma_cfg      = {.hartid = hartid};
    idma_controller_t idma_ctrl = {.base = NULL, .cfg = &idma_cfg, .api = &idma_api};
    idma_init(&idma_ctrl);

    fsync_config_t fsync_cfg      = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {.base = NULL, .cfg = &fsync_cfg, .api = &fsync_api};
    fsync_init(&fsync_ctrl);

#if STALLING == 0
    eu_config_t eu_cfg      = {.hartid = hartid};
    eu_controller_t eu_ctrl = {.base = NULL, .cfg = &eu_cfg, .api = &eu_api};
    eu_init(&eu_ctrl);
    eu_clear_events(0xFFFFFFFF);
    eu_fsync_init(&eu_ctrl, 0);
    eu_idma_init(&eu_ctrl, 0);
    eu_controller_t *eu = &eu_ctrl;
#else
    eu_controller_t *eu = NULL;
#endif

    // Consumer gives one sub-ring per possible producer (keyed by hartid) and
    // sizes each to hold every producing case (no slot reuse, so fifo_pop's
    // zero-copy payload stays valid); every other tile initializes an empty FIFO.
    // Done before the startup barrier so head/tail are reset before the producer
    // pushes.
    uint32_t num_producers = (hartid == FIFO_SLICE_CONSUMER) ? NUM_TILES : 0u;
    uint32_t num_slots     = (hartid == FIFO_SLICE_CONSUMER) ? FIFO_SLICE_NUM_CASES : 0u;
    fifo_init(hartid, num_producers, num_slots, FIFO_SLICE_SLOT_BYTES);

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    uint32_t errors = 0u;

    if (hartid == FIFO_SLICE_PRODUCER) {
        uint32_t buf = get_l1_base(hartid) + 0x1000u; // clear of our own FIFO header

        for (uint32_t c = 0; c < FIFO_SLICE_NUM_CASES; ++c) {
            const slice_case_t *cs = &CASES[c];

            // Sentinel the buffer, then stamp the pattern into the source
            // element slots (honoring the src descriptor's strides/starts).
            fill_bytes(buf, SENTINEL, FIFO_SLICE_SRC_BUF_BYTES);
            for (uint32_t k = 0; k < cs->src.num_elems; ++k) {
                uint32_t so = cs->src_off + desc_off(&cs->src, k);
                for (uint32_t j = 0; j < cs->elem_bytes; ++j)
                    poke8(buf + so + j, pattern_byte(c, k, j));
            }

            fifo_push_req_t req = {
                .target_hartid = FIFO_SLICE_CONSUMER,
                .producer_idx  = hartid,
                .src_base_addr = buf + cs->src_off,
                .src           = &cs->src,
                .desc          = &cs->desc,
                .tag           = c,
                .elem_bytes    = cs->elem_bytes,
            };
            if (fifo_push(&idma_ctrl, eu, &req) != 0) {
                printf("[fifo_slice] push failed for case %u (%s)\n", c, cs->name);
                errors++;
            }
        }

        // Negative case: a source rank beyond IDMA_ND_MAX_RANK must be rejected
        // (returns -1) without publishing or touching the out-of-range dim.
        tensor_sub_slice_t bad_src = {.rank = IDMA_ND_MAX_RANK + 1u, .num_elems = 1u};
        fifo_push_req_t bad_req    = {
               .target_hartid = FIFO_SLICE_CONSUMER,
               .producer_idx  = hartid,
               .src_base_addr = buf,
               .src           = &bad_src,
               .desc          = &bad_src,
               .tag           = 0xBADu,
               .elem_bytes    = 2u,
        };
        if (fifo_push(&idma_ctrl, eu, &bad_req) != -1) {
            printf("[fifo_slice] over-rank push was not rejected\n");
            errors++;
        }
    }

    if (hartid == FIFO_SLICE_CONSUMER) {
        uint8_t seen[FIFO_SLICE_NUM_CASES];
        for (uint32_t i = 0; i < FIFO_SLICE_NUM_CASES; ++i)
            seen[i] = 0u;

        uint32_t remaining = FIFO_SLICE_NUM_CASES;
        while (remaining > 0u) {
            fifo_msg_t msg;
            while (!fifo_pop(hartid, &msg))
                ;

            uint32_t c = msg.tag;
            if (c >= FIFO_SLICE_NUM_CASES || seen[c]) {
                printf("[fifo_slice] unexpected/duplicate tag %u\n", c);
                errors++;
                continue;
            }
            seen[c] = 1u;
            remaining--;

            uint32_t case_errs = verify_case(c, &msg);
            printf("[%s] errors: %u\n", CASES[c].name, case_errs);
            errors += case_errs;
        }
    }

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    if (hartid == FIFO_SLICE_CONSUMER || hartid == FIFO_SLICE_PRODUCER)
        printf("Tile %u fifo_slice errors: %u\n", hartid, errors);

    return errors;
}
