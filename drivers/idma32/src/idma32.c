// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Alberto Dequino <alberto.dequino@unibo.it>
//
// This file provides the strong (driver-specific) implementations for the
// IDMA functions using a 32-bit representation (a struct with 'low' and 'high').
// These functions override the weak HAL symbols.
// This is a WIP and might be redundant, as the moment of writing there is only one IDMA
// configuration tested on MAGIA.

#include <stdint.h>
#include "idma32.h"
#include "regs/tile_ctrl.h"
#include "utils/idma_isa_utils.h"
#include "utils/magia_utils.h"
// #include "utils/tinyprintf.h"
#include "utils/printf.h"

int idma32_init(idma_controller_t *ctrl)
{
    uint32_t index = (1 << IRQ_A2O_DONE) | (1 << IRQ_O2A_DONE);
    irq_en(index);
    return 0;
}

/* static inline __attribute__((always_inline)) void idma32_wait(){
    asm volatile("wfi" ::: "memory");
} */

/**
 * Start 1-dimensional memory copy
 *
 * @param dir Copy Direction. 0 = AXI to OBI (L2 to L1), !0 = OBI to AXI (L1 to L2).
 * @param axi_addr AXI/L2 memory address of first element.
 * @param obi_addr OBI/L1 memory address of first element.
 * @param len Byte length of memory block to transfer.
 */
int idma32_memcpy_1d(
    idma_controller_t *ctrl, uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len)
{
#if IDMA_MM == 0
    if (dir) { // OBI to AXI (L1 to L2)
        idma_conf_out();
        idma_set_addr_len_out(axi_addr, obi_addr, len);
        idma_set_std2_rep2_out(0, 0, 1);
        idma_set_std3_rep3_out(0, 0, 1);
        idma_start_out();
        // printf("IDMA_memcpy_1d: Detected IRQ...\n");
    } else { // AXI to OBI (L2 to L1)
        idma_conf_in();
        idma_set_addr_len_in(obi_addr, axi_addr, len);
        idma_set_std2_rep2_in(0, 0, 1);
        idma_set_std3_rep3_in(0, 0, 1);
        idma_start_in();
        // printf("IDMA_memcpy_1d: Detected IRQ...\n");
    }
    return 0;
#else
    idma_mm_conf(dir, 0, 0, 0, 0, 0, 0, 3);
    if (dir) {
        idma_mm_set_addr_len(dir, axi_addr, obi_addr, len);
    } else {
        idma_mm_set_addr_len(dir, obi_addr, axi_addr, len);
    }
    idma_mm_set_std2_rep2(dir, 0, 0, 1);
    idma_mm_set_std3_rep3(dir, 0, 0, 1);
    idma_mm_start(dir);
#endif
}

/**
 * Start 2-dimensional memory copy.
 *
 * @param dir Copy Direction. 0 = AXI to OBI (L2 to L1), !0 = OBI to AXI (L1 to L2).
 * @param axi_addr AXI/L2 memory address of first element.
 * @param obi_addr OBI/L1 memory address of first element.
 * @param len Byte length of memory block to transfer for each repetition.
 * @param std Offset to add to the memory address (axi_addr or obi_addr) to calculate the start of
 * the next memory block.
 * @param reps Number of repetitions.
 */
int idma32_memcpy_2d(idma_controller_t *ctrl,
                     uint8_t dir,
                     uint32_t axi_addr,
                     uint32_t obi_addr,
                     uint32_t len,
                     uint32_t std,
                     uint32_t reps)
{
// printf("IDMA Transfer! Direction: %d\n", dir);
#if IDMA_MM == 0
    if (dir) { // OBI to AXI (L1 to L2)
        idma_conf_out();
        idma_set_addr_len_out(axi_addr, obi_addr, len);
        idma_set_std2_rep2_out(std, len, reps);
        idma_set_std3_rep3_out(0, 0, 1);
        idma_start_out();
        // printf("IDMA_memcpy_2d: Detected IRQ...\n");
    } else { // AXI to OBI (L2 to L1)
        idma_conf_in();
        idma_set_addr_len_in(obi_addr, axi_addr, len);
        idma_set_std2_rep2_in(len, std, reps);
        idma_set_std3_rep3_in(0, 0, 1);
        idma_start_in();
        // printf("IDMA_memcpy_2d: Detected IRQ...\n");
    }
#else
    // printf("IDMA Transfer! Direction: %d\n", dir);
    idma_mm_conf(dir, 0, 0, 0, 0, 0, 0, 3);
    if (dir) {
        idma_mm_set_addr_len(dir, axi_addr, obi_addr, len);
        idma_mm_set_std2_rep2(dir, std, len, reps);
    } else {
        idma_mm_set_addr_len(dir, obi_addr, axi_addr, len);
        idma_mm_set_std2_rep2(dir, len, std, reps);
    }
    idma_mm_set_std3_rep3(dir, 0, 0, 1);
    idma_mm_start(dir);
#endif
    return 0;
}

/**
 * Start 3-dimensional memory copy.
 *
 * @param dir Copy Direction. 0 = AXI to OBI (L2 to L1), !0 = OBI to AXI (L1 to L2).
 * @param axi_addr AXI/L2 memory address of first element.
 * @param obi_addr OBI/L1 memory address of first element.
 * @param len Byte length of each row to transfer.
 * @param std2 Stride between row starts on the strided (L2/AXI) side (2nd dimension).
 * @param reps2 Number of rows per plane (repetitions of the 2nd dimension).
 * @param std3 Stride between plane starts on the strided (L2/AXI) side (3rd dimension).
 * @param reps3 Number of planes (repetitions of the 3rd dimension).
 */
int idma32_memcpy_3d(idma_controller_t *ctrl,
                     uint8_t dir,
                     uint32_t axi_addr,
                     uint32_t obi_addr,
                     uint32_t len,
                     uint32_t std2,
                     uint32_t reps2,
                     uint32_t std3,
                     uint32_t reps3)
{
#if IDMA_MM == 0
    if (dir) { // OBI to AXI (L1 to L2)
        idma_conf_out();
        idma_set_addr_len_out(axi_addr, obi_addr, len);
        idma_set_std2_rep2_out(std2, len, reps2);
        idma_set_std3_rep3_out(std3, len * reps2, reps3);
        idma_start_out();
    } else { // AXI to OBI (L2 to L1)
        idma_conf_in();
        idma_set_addr_len_in(obi_addr, axi_addr, len);
        idma_set_std2_rep2_in(len, std2, reps2);
        idma_set_std3_rep3_in(len * reps2, std3, reps3);
        idma_start_in();
    }
#else
    idma_mm_conf(dir, 0, 0, 0, 0, 0, 0, 3);
    if (dir) {
        idma_mm_set_addr_len(dir, axi_addr, obi_addr, len);
        idma_mm_set_std2_rep2(dir, std2, len, reps2);
        idma_mm_set_std3_rep3(dir, std3, len * reps2, reps3);
    } else {
        idma_mm_set_addr_len(dir, obi_addr, axi_addr, len);
        idma_mm_set_std2_rep2(dir, len, std2, reps2);
        idma_mm_set_std3_rep3(dir, len * reps2, std3, reps3);
    }
    idma_mm_start(dir);
#endif
    return 0;
}

/**
 * Normalized (canonical) form of one side of a copy: per-dim lengths/strides
 * after folding per-dim `start` offsets into a base byte offset, dropping
 * singleton (length-1) dims, and coalescing contiguous adjacent dims. Dims are
 * kept outermost-first (row-major): dims[0] varies slowest, dims[rank-1] fastest.
 */
typedef struct {
    uint32_t rank;
    uint32_t len[IDMA_ND_MAX_RANK];
    uint32_t std[IDMA_ND_MAX_RANK];
    uint32_t base_off; /**< Byte offset to add to the side's base address. */
} nd_norm_t;

/**
 * Greatest common divisor (Euclidean). Used to size the 1D-fallback burst: the
 * largest block of consecutive elements that stays contiguous on both sides is
 * the gcd of the two sides' innermost contiguous run lengths, since that block
 * must divide both runs to tile them on aligned, regularly-spaced boundaries.
 */
static uint32_t nd_gcd(uint32_t a, uint32_t b)
{
    while (b != 0u) {
        uint32_t t = a % b;
        a          = b;
        b          = t;
    }
    return a;
}

/* Fold starts into a base offset, drop length-1 dims, then coalesce adjacent
 * dims where the outer stride equals inner_length * inner_stride. */
static void nd_normalize(const tensor_sub_slice_t *s, nd_norm_t *out)
{
    uint32_t base_off = 0u;
    out->rank         = 0u;

    // Walk the input dims outermost-first, building the canonical form in `out`.
    for (uint32_t d = 0; d < s->rank; ++d) {
        // Step 1: fold this dim's start into a constant base offset. The start is
        // a fixed index shift, so it never affects the iteration counts/strides;
        // accumulating start*stride here lets the rest of the pass ignore starts.
        base_off += s->dims[d].start * s->dims[d].stride;

        // Step 2: drop singleton dims. A length-1 dim spans a single element, so
        // it adds nothing to the walk (its start was already folded above).
        if (s->dims[d].length == 1u) {
            continue;
        }

        uint32_t len = s->dims[d].length;
        uint32_t std = s->dims[d].stride;

        // Step 3: coalesce with the previous (more-outer) kept dim when the two
        // are perfectly nested in memory, i.e. the outer stride exactly spans the
        // inner dim (outer_stride == inner_length * inner_stride). Merging keeps
        // the inner stride and multiplies the lengths, collapsing a contiguous
        // block into one dim so later stages see the widest possible runs.
        if (out->rank > 0u && out->std[out->rank - 1u] == len * std) {
            out->len[out->rank - 1u] *= len;
            out->std[out->rank - 1u] = std;
        } else {
            // Not mergeable: append it as a new (more-inner) dim.
            out->len[out->rank] = len;
            out->std[out->rank] = std;
            out->rank++;
        }
    }

    out->base_off = base_off;
}

/* Byte address of row-major linear element k for a normalized side. This decodes
 * the flat index k into per-dim indices (mixed-radix, radices = dim lengths) and
 * accumulates each index * stride onto the base. */
static uint32_t nd_addr(uint32_t base, const nd_norm_t *n, uint32_t k)
{
    uint32_t addr = base;

    // Walk innermost-first (dims[rank-1] is the fastest-varying, row-major).
    for (uint32_t d = n->rank; d-- > 0u;) {
        // Peel off this dim's index: the remainder mod its length is the index
        // along dim d, and the quotient carries the higher (more-outer) dims.
        uint32_t idx = k % n->len[d];
        k /= n->len[d];

        // Advance the address by that many strides along this dim.
        addr += idx * n->std[d];
    }
    return addr;
}

static inline void nd_wait(eu_controller_t *eu_ctrl, uint8_t dir)
{
    if (eu_ctrl != NULL) {
        if (dir == 0u) {
            eu_idma_wait_a2o(eu_ctrl, IDMA_ND_WAIT_MODE);
        } else {
            eu_idma_wait_o2a(eu_ctrl, IDMA_ND_WAIT_MODE);
        }
    }
}

/* True when a normalized side is a single packed run (or a single element). */
static inline int nd_is_contiguous(const nd_norm_t *n, uint32_t elem_bytes)
{
    return (n->rank == 0u) || (n->rank == 1u && n->std[0] == elem_bytes);
}

/**
 * Dispatch a multi-dimensional copy between two independently-shaped sub-slices
 * to the widest native IDMA burst that fits (2D or 1D), decided at runtime.
 *
 * The two ports involved are OBI, wired only to this tile's own L1, and AXI, the
 * mesh port that reaches either L2 or any other tile's L1 over the NoC. So the
 * same routine serves L2<->L1 transfers and tile-to-tile L1->L1 transfers; the
 * AXI endpoint is just an address, its physical target (L2 or remote L1) does not
 * matter here. Only the local (OBI) side can never be strided by the hardware.
 *
 * Both sides are normalized (starts folded, singleton dims dropped, contiguous
 * dims coalesced). When the OBI side is a single packed run, the transfer is
 * issued as native 2D bursts over the strided (AXI) side's dims — collapsing to a
 * single 1D burst when both sides are fully contiguous. When the OBI side is not
 * contiguous, it falls back to a software loop of 1D bursts over the largest
 * common contiguous run. The controller carries only one in-flight transfer, so
 * each burst is waited on via eu_ctrl (if non-NULL) before the next.
 *
 * @param dir Copy Direction. 0 = AXI to OBI (remote/L2 -> local L1), !0 = OBI to
 *            AXI (local L1 -> remote/L2).
 * @param dst_addr Destination address of first element.
 * @param src_addr Source address of first element.
 * @param src Source sub-slice descriptor.
 * @param dst Destination sub-slice descriptor.
 * @param elem_bytes Size of a single element in bytes (shared by both sides).
 * @param eu_ctrl Event-unit controller used to wait for each burst; pass NULL to skip waiting.
 */
int idma32_memcpy_md_to_nd(idma_controller_t *ctrl,
                           uint8_t dir,
                           uint32_t dst_addr,
                           uint32_t src_addr,
                           const tensor_sub_slice_t *src,
                           const tensor_sub_slice_t *dst,
                           uint32_t elem_bytes,
                           eu_controller_t *eu_ctrl)
{
    // Step 0: reject/short-circuit the degenerate cases. Ranks must fit the
    // fixed working arrays, the two sides must describe the same element count
    // (they are copied element-for-element), and an empty transfer is a no-op.
    if (src->rank > IDMA_ND_MAX_RANK || dst->rank > IDMA_ND_MAX_RANK) {
        return -1;
    }
    if (src->num_elems != dst->num_elems) {
        return -1;
    }
    if (src->num_elems == 0u) {
        return 0;
    }

    // Step 1: normalize each side to canonical form (starts folded into a base
    // offset, singleton dims dropped, contiguous dims coalesced), then form each
    // side's real starting address by adding its folded offset to the base addr.
    nd_norm_t src_n;
    nd_norm_t dst_n;
    nd_normalize(src, &src_n);
    nd_normalize(dst, &dst_n);

    uint32_t total         = src->num_elems;
    uint32_t src_full_base = src_addr + src_n.base_off;
    uint32_t dst_full_base = dst_addr + dst_n.base_off;

    // Step 2: bind the physical AXI (freely strided; L2 or a remote tile's L1)
    // and OBI (this tile's local L1) ports to src/dst according to direction.
    // dir==0 pulls remote->local (src is AXI, dst is OBI); dir!=0 pushes
    // local->remote (src is OBI, dst is AXI). "Remote" is L2 or another tile's L1.
    const nd_norm_t *axi_n = (dir == 0u) ? &src_n : &dst_n;
    const nd_norm_t *obi_n = (dir == 0u) ? &dst_n : &src_n;
    uint32_t axi_base      = (dir == 0u) ? src_full_base : dst_full_base;
    uint32_t obi_base      = (dir == 0u) ? dst_full_base : src_full_base;

    // Step 3: choose a path from the OBI side. Native 2D/1D bursts require the
    // OBI (L1) side to be a single packed run, since the hardware only strides
    // the AXI side; if it is, take the fast path, otherwise fall through to the
    // software 1D loop that can handle arbitrary strides on both sides.
    if (nd_is_contiguous(obi_n, elem_bytes)) {
        // Step 3a: pick the burst block. Peel the AXI side's innermost dim into
        // the contiguous chunk copied per repetition: if that dim is itself
        // packed (stride == elem_bytes) the block is its whole run and the
        // remaining outer dims are the strided levels (sr); otherwise the block
        // is a single element and every AXI dim is a strided level.
        uint32_t block_bytes;
        uint32_t sr; // number of strided levels
        const uint32_t *slen;
        const uint32_t *sstd;

        if (axi_n->rank >= 1u && axi_n->std[axi_n->rank - 1u] == elem_bytes) {
            block_bytes = axi_n->len[axi_n->rank - 1u] * elem_bytes;
            sr          = axi_n->rank - 1u;
        } else {
            block_bytes = elem_bytes;
            sr          = axi_n->rank;
        }
        slen = axi_n->len;
        sstd = axi_n->std;

        // Step 3b: with no strided levels left, the AXI side is also fully packed
        // — the whole transfer is one contiguous 1D burst.
        if (sr == 0u) {
            idma32_memcpy_1d(ctrl, dir, axi_base, obi_base, total * elem_bytes);
            nd_wait(eu_ctrl, dir);
            return 0;
        }

        // Step 3c: the innermost remaining strided dim supplies the 2D burst's
        // (stride, reps); any dims outside it are iterated by a software odometer,
        // one 2D burst per outer position.
        uint32_t reps       = slen[sr - 1u];
        uint32_t std        = sstd[sr - 1u];
        uint32_t outer_rank = sr - 1u;
        uint32_t coord[IDMA_ND_MAX_RANK] = {0u, 0u, 0u, 0u};
        uint32_t obi_cursor              = obi_base;

        for (;;) {
            // Offset the AXI base by the current outer-dim coordinates. The OBI
            // side is packed, so its cursor just advances linearly per burst.
            uint32_t axi_block = axi_base;
            for (uint32_t i = 0; i < outer_rank; ++i) {
                axi_block += coord[i] * sstd[i];
            }

            idma32_memcpy_2d(ctrl, dir, axi_block, obi_cursor, block_bytes, std, reps);
            nd_wait(eu_ctrl, dir);

            obi_cursor += reps * block_bytes;

            // A single strided level means one 2D burst covered everything.
            if (outer_rank == 0u) {
                break;
            }

            // Advance the odometer over the outer dims (innermost-first, with
            // carry); when it wraps past the outermost dim, all bursts are done.
            uint32_t carried = 1u;
            for (uint32_t d = outer_rank; d-- > 0u;) {
                coord[d]++;
                if (coord[d] < slen[d]) {
                    carried = 0u;
                    break;
                }
                coord[d] = 0u;
            }
            if (carried) {
                break;
            }
        }

        return 0;
    }

    // Step 4 (fallback): the OBI side is strided too, so no native 2D burst
    // applies. Copy in 1D bursts of the largest run that is contiguous on both
    // sides at once.
    //
    // Step 4a: each side's innermost contiguous run is its innermost dim length
    // when that dim is packed, else a single element; their gcd is the largest
    // block that tiles both sides on aligned boundaries.
    uint32_t src_contig =
        (src_n.rank >= 1u && src_n.std[src_n.rank - 1u] == elem_bytes) ? src_n.len[src_n.rank - 1u]
                                                                       : 1u;
    uint32_t dst_contig =
        (dst_n.rank >= 1u && dst_n.std[dst_n.rank - 1u] == elem_bytes) ? dst_n.len[dst_n.rank - 1u]
                                                                       : 1u;
    uint32_t block_elems = nd_gcd(src_contig, dst_contig);
    uint32_t block_bytes = block_elems * elem_bytes;

    // Step 4b: step through the flat element space one block at a time, decoding
    // each side's address for the block's first element (nd_addr) and issuing a
    // 1D burst with the AXI address passed first, as the HAL expects.
    for (uint32_t k = 0; k < total; k += block_elems) {
        uint32_t sa = nd_addr(src_full_base, &src_n, k);
        uint32_t da = nd_addr(dst_full_base, &dst_n, k);

        if (dir == 0u) {
            idma32_memcpy_1d(ctrl, dir, sa, da, block_bytes);
        } else {
            idma32_memcpy_1d(ctrl, dir, da, sa, block_bytes);
        }
        nd_wait(eu_ctrl, dir);
    }

    return 0;
}

extern int idma_init(idma_controller_t *ctrl)
    __attribute__((alias("idma32_init"), used, visibility("default")));
/* extern void idma_wait()
    __attribute__((alias("idma32_wait"), used, visibility("default"))); */
extern int idma_memcpy_1d(
    idma_controller_t *ctrl, uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len)
    __attribute__((alias("idma32_memcpy_1d"), used, visibility("default")));
extern int idma_memcpy_2d(idma_controller_t *ctrl,
                          uint8_t dir,
                          uint32_t axi_addr,
                          uint32_t obi_addr,
                          uint32_t len,
                          uint32_t std,
                          uint32_t reps)
    __attribute__((alias("idma32_memcpy_2d"), used, visibility("default")));
extern int idma_memcpy_3d(idma_controller_t *ctrl,
                          uint8_t dir,
                          uint32_t axi_addr,
                          uint32_t obi_addr,
                          uint32_t len,
                          uint32_t std2,
                          uint32_t reps2,
                          uint32_t std3,
                          uint32_t reps3)
    __attribute__((alias("idma32_memcpy_3d"), used, visibility("default")));
extern int idma_memcpy_md_to_nd(idma_controller_t *ctrl,
                                uint8_t dir,
                                uint32_t dst_addr,
                                uint32_t src_addr,
                                const tensor_sub_slice_t *src,
                                const tensor_sub_slice_t *dst,
                                uint32_t elem_bytes,
                                eu_controller_t *eu_ctrl)
    __attribute__((alias("idma32_memcpy_md_to_nd"), used, visibility("default")));

/* Export the IDMA-specific controller API */
idma_controller_api_t idma_api = {
    .init = idma32_init,
    /*     .wait = idma32_wait, */
    .memcpy_1d       = idma32_memcpy_1d,
    .memcpy_2d       = idma32_memcpy_2d,
    .memcpy_3d       = idma32_memcpy_3d,
    .memcpy_md_to_nd = idma32_memcpy_md_to_nd,
};