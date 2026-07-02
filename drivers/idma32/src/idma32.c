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
 * Dispatch an n-dimensional memory copy to the narrowest native IDMA burst that
 * fits it (1D, 2D, or 3D), decided at runtime from copy's shape/strides after
 * coalescing contiguous innermost dimensions.
 *
 * TODO: dispatch to a native 2D burst (idma32_memcpy_2d) when the transfer
 * reduces to a single (std, rep) level, and to a native 3D burst (using the
 * controller's std3/rep3 registers) when it reduces to two levels. For now,
 * only the fully-contiguous case is handled natively (as a single 1D burst);
 * anything else prints a warning and falls back to a software loop of 1D
 * bursts, waiting on eu_ctrl (if non-NULL) between each.
 *
 * @param dir Copy Direction. 0 = AXI to OBI (L2 to L1), !0 = OBI to AXI (L1 to L2).
 * @param dst_addr Destination address of first element.
 * @param src_addr Source address of first element.
 * @param copy Copy descriptor (rank, shape, element size, per-side strides).
 * @param eu_ctrl Event-unit controller used to wait for each burst; pass NULL to skip waiting.
 */
int idma32_memcpy_md_to_nd(idma_controller_t *ctrl,
                           uint8_t dir,
                           uint32_t dst_addr,
                           uint32_t src_addr,
                           const copy_desc_t *copy,
                           eu_controller_t *eu_ctrl)
{
    uint32_t coord[IDMA_ND_MAX_RANK] = {0u, 0u, 0u, 0u};
    uint32_t iter_rank;
    uint32_t inner_dim;
    uint32_t burst_bytes;

    if (copy->rank == 0u) {
        return 0;
    }

    if (copy->rank > IDMA_ND_MAX_RANK) {
        return -1;
    }

    for (uint32_t d = 0; d < copy->rank; ++d) {
        if (copy->shape[d] == 0u) {
            return 0;
        }
    }

    inner_dim = copy->rank - 1u;
    if (copy->src_strides_bytes[inner_dim] == copy->elem_bytes &&
        copy->dst_strides_bytes[inner_dim] == copy->elem_bytes) {
        iter_rank   = inner_dim;
        burst_bytes = copy->shape[inner_dim] * copy->elem_bytes;
    } else {
        iter_rank   = copy->rank;
        burst_bytes = copy->elem_bytes;
    }

    if (iter_rank == 0u) {
        // Fully contiguous: a single native 1D burst covers the whole transfer.
        int ret;

        if (dir == 0u) {
            ret = idma32_memcpy_1d(ctrl, dir, src_addr, dst_addr, burst_bytes);
        } else {
            ret = idma32_memcpy_1d(ctrl, dir, dst_addr, src_addr, burst_bytes);
        }

        if (eu_ctrl != NULL) {
            if (dir == 0u) {
                eu_idma_wait_a2o(eu_ctrl, IDMA_ND_WAIT_MODE);
            } else {
                eu_idma_wait_o2a(eu_ctrl, IDMA_ND_WAIT_MODE);
            }
        }

        return ret;
    }

    printf("Warning: idma32_memcpy_md_to_nd: no native %uD burst yet, falling back to 1D bursts\n",
           iter_rank + 1u);

    for (;;) {
        uint32_t src_block_addr = src_addr;
        uint32_t dst_block_addr = dst_addr;

        for (uint32_t d = 0; d < iter_rank; ++d) {
            src_block_addr += coord[d] * copy->src_strides_bytes[d];
            dst_block_addr += coord[d] * copy->dst_strides_bytes[d];
        }

        if (dir == 0u) {
            idma32_memcpy_1d(ctrl, dir, src_block_addr, dst_block_addr, burst_bytes);
        } else {
            idma32_memcpy_1d(ctrl, dir, dst_block_addr, src_block_addr, burst_bytes);
        }

        if (eu_ctrl != NULL) {
            if (dir == 0u) {
                eu_idma_wait_a2o(eu_ctrl, IDMA_ND_WAIT_MODE);
            } else {
                eu_idma_wait_o2a(eu_ctrl, IDMA_ND_WAIT_MODE);
            }
        }

        for (uint32_t d = iter_rank; d > 0u; --d) {
            uint32_t idx = d - 1u;

            coord[idx]++;
            if (coord[idx] < copy->shape[idx]) {
                break;
            }

            coord[idx] = 0u;
            if (idx == 0u) {
                return 0;
            }
        }
    }
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
                                const copy_desc_t *copy,
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