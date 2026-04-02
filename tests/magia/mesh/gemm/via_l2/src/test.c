// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "fsync.h"
#include "idma.h"
#include "redmule.h"
#include "eventunit.h"

#define WAIT_MODE WFE

/**
 * Tile group definitions for 8x8 mesh (64 tiles).
 *
 * GEMM1 tiles: [0, 1, 8, 9]           (4 tiles,  top-left 2x2 block)
 * GEMM2 tiles: [16-19, 24-27, 32-35,
 *               40-43, 48-51, 56-59]   (24 tiles, rows 2-7 cols 0-3)
 * GEMM3 tiles: [2-7, 10-15]           (12 tiles, rows 0-1 cols 2-7)
 * GEMM4 tiles: [20-23, 28-31, 36-39,
 *               44-47, 52-55, 60-63]   (24 tiles, rows 2-7 cols 4-7)
 */
#define GEMM1_N_TILES 4
#define GEMM2_N_TILES 24
#define GEMM3_N_TILES 12
#define GEMM4_N_TILES 24

#define abs_threshold_millis 8 /* 0.008 expressed as integer millis */

static const uint32_t gemm1_tiles[GEMM1_N_TILES] = {0, 1, 8, 9};

static const uint32_t gemm2_tiles[GEMM2_N_TILES] = {16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35,
                                                    40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59};

static const uint32_t gemm3_tiles[GEMM3_N_TILES] = {2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15};

static const uint32_t gemm4_tiles[GEMM4_N_TILES] = {20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39,
                                                    44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63};

int get_local_idx(uint32_t hartid, const uint32_t *tiles, uint32_t n_tiles)
{
    for (uint32_t i = 0; i < n_tiles; i++)
        if (tiles[i] == hartid)
            return (int)i;
    return -1;
}

void get_row_range(uint32_t local_idx,
                   uint32_t n_tiles,
                   uint32_t total_rows,
                   uint32_t *start_row,
                   uint32_t *num_rows)
{
    uint32_t base = total_rows / n_tiles;
    uint32_t rem  = total_rows % n_tiles;
    *start_row    = local_idx * base + (local_idx < rem ? local_idx : rem);
    *num_rows     = base + (local_idx < rem ? 1 : 0);
}

int mem_set_zero(uint32_t o, uint32_t dim)
{
    for (uint32_t i = 0; i < dim; i++)
        mmio16(o + i * 2) = 0x0000;
}

/**
 * GEMM chain test with row-parallel data parallelism across tile groups.
 *
 * Phase 1 (parallel):
 *   GEMM1 group (4 tiles):  R1[AxC] = M1[AxB] @ M2[BxC]
 *   GEMM2 group (24 tiles): R2[CxE] = M3[CxD] @ M4[DxE]
 *
 * Phase 2:
 *   GEMM3 group (12 tiles): R3[AxE] = R1[AxC] @ R2[CxE]
 *
 * Phase 3:
 *   GEMM4 group (24 tiles): O[AxF]  = R3[AxE] @ M5[ExF]
 *
 * Each group splits output rows across its tiles. Tiles with no rows idle.
 * Requires 8x8 mesh (64 tiles).
 */
int main(void)
{
    /**
     * 0. Initializations
     */
    uint32_t hartid       = get_hartid();
    uint32_t x_id         = GET_X_ID(hartid);
    uint32_t y_id         = GET_Y_ID(hartid);
    uint32_t l1_tile_base = get_l1_base(hartid);

    // Init iDMA
    idma_config_t idma_cfg      = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg  = &idma_cfg,
        .api  = &idma_api,
    };
    idma_init(&idma_ctrl);

    // Init RedMulE
    redmule_config_t redmule_cfg      = {.hartid = hartid};
    redmule_controller_t redmule_ctrl = {
        .base = NULL,
        .cfg  = &redmule_cfg,
        .api  = &redmule_api,
    };
    redmule_init(&redmule_ctrl);

    // Init FractalSync
    fsync_config_t fsync_cfg      = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };
    fsync_init(&fsync_ctrl);

// Init the Event Unit controller
#if STALLING == 0
    eu_config_t eu_cfg      = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg  = &eu_cfg,
        .api  = &eu_api,
    };

    eu_init(&eu_ctrl);
    eu_clear_events(0xFFFFFFFF);
    eu_fsync_init(&eu_ctrl, 0);
    eu_idma_init(&eu_ctrl, 0);
    eu_redmule_init(&eu_ctrl, 0);
#endif

    // Global barrier: ensure all tiles have completed startup (including BSS zeroing
    // in crt0.S) before any tile begins writing results to L2 output buffers.
    // Without this, slow tiles still zeroing BSS can overwrite Phase 1 results.
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /**
     * Phase 1: GEMM1 and GEMM2 in parallel (row-parallel within each group)
     *   GEMM1 group: R1 = M1 @ M2
     *   GEMM2 group: R2 = M3 @ M4
     */
    int gemm1_idx = get_local_idx(hartid, gemm1_tiles, GEMM1_N_TILES);
    if (gemm1_idx >= 0) {
        uint32_t start_row, num_rows;
        get_row_range(gemm1_idx, GEMM1_N_TILES, DIM_A, &start_row, &num_rows);

        if (num_rows > 0) {
            // L1 layout: M1_slice, M2, R1_slice
            uint32_t obi_m1 = l1_tile_base;
            uint32_t obi_m2 = obi_m1 + (num_rows * DIM_B * 2);
            uint32_t obi_r1 = obi_m2 + (DIM_B * DIM_C * 2);

            // Load slice of M1 [num_rows x B] from L2
            idma_memcpy_1d(&idma_ctrl,
                           0,
                           (uint32_t)m1_inp + start_row * DIM_B * 2,
                           obi_m1,
                           num_rows * DIM_B * 2);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

            // Load full M2 [BxC] from L2
            idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m2_inp, obi_m2, DIM_B * DIM_C * 2);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

            // Zero accumulator and compute: R1_slice = M1_slice @ M2
            mem_set_zero(obi_r1, num_rows * DIM_C);
            redmule_gemm(&redmule_ctrl,
                         obi_m1,
                         obi_m2,
                         obi_r1,
                         (uint16_t)num_rows,
                         (uint16_t)DIM_B,
                         (uint16_t)DIM_C);
            eu_redmule_wait(&eu_ctrl, WAIT_MODE);

            // Write R1 slice back to L2 at correct offset
            idma_memcpy_1d(&idma_ctrl,
                           1,
                           (uint32_t)r1_out + start_row * DIM_C * 2,
                           obi_r1,
                           num_rows * DIM_C * 2);
            eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
        }
    }

    int gemm2_idx = get_local_idx(hartid, gemm2_tiles, GEMM2_N_TILES);
    if (gemm2_idx >= 0) {
        uint32_t start_row, num_rows;
        get_row_range(gemm2_idx, GEMM2_N_TILES, DIM_C, &start_row, &num_rows);

        if (num_rows > 0) {
            // L1 layout: M3_slice, M4, R2_slice
            uint32_t obi_m3 = l1_tile_base;
            uint32_t obi_m4 = obi_m3 + (num_rows * DIM_D * 2);
            uint32_t obi_r2 = obi_m4 + (DIM_D * DIM_E * 2);

            // Load slice of M3 [num_rows x D] from L2
            idma_memcpy_1d(&idma_ctrl,
                           0,
                           (uint32_t)m3_inp + start_row * DIM_D * 2,
                           obi_m3,
                           num_rows * DIM_D * 2);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

            // Load full M4 [DxE] from L2
            idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m4_inp, obi_m4, DIM_D * DIM_E * 2);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

            // Zero accumulator and compute: R2_slice = M3_slice @ M4
            mem_set_zero(obi_r2, num_rows * DIM_E);
            redmule_gemm(&redmule_ctrl,
                         obi_m3,
                         obi_m4,
                         obi_r2,
                         (uint16_t)num_rows,
                         (uint16_t)DIM_D,
                         (uint16_t)DIM_E);
            eu_redmule_wait(&eu_ctrl, WAIT_MODE);

            // Write R2 slice back to L2 at correct offset
            idma_memcpy_1d(&idma_ctrl,
                           1,
                           (uint32_t)r2_out + start_row * DIM_E * 2,
                           obi_r2,
                           num_rows * DIM_E * 2);
            eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
        }
    }

    // Global barrier: wait for Phase 1 to complete
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /**
     * Phase 2: GEMM3 (row-parallel)
     *   GEMM3 group: R3 = R1 @ R2
     */
    int gemm3_idx = get_local_idx(hartid, gemm3_tiles, GEMM3_N_TILES);
    if (gemm3_idx >= 0) {
        uint32_t start_row, num_rows;
        get_row_range(gemm3_idx, GEMM3_N_TILES, DIM_A, &start_row, &num_rows);

        if (num_rows > 0) {
            // L1 layout: R1_slice, R2, R3_slice
            uint32_t obi_r1 = l1_tile_base;
            uint32_t obi_r2 = obi_r1 + (num_rows * DIM_C * 2);
            uint32_t obi_r3 = obi_r2 + (DIM_C * DIM_E * 2);

            // Load slice of R1 [num_rows x C] from L2
            idma_memcpy_1d(&idma_ctrl,
                           0,
                           (uint32_t)r1_out + start_row * DIM_C * 2,
                           obi_r1,
                           num_rows * DIM_C * 2);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

            // Load full R2 [CxE] from L2
            idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)r2_out, obi_r2, DIM_C * DIM_E * 2);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

            // Zero accumulator and compute: R3_slice = R1_slice @ R2
            mem_set_zero(obi_r3, num_rows * DIM_E);
            redmule_gemm(&redmule_ctrl,
                         obi_r1,
                         obi_r2,
                         obi_r3,
                         (uint16_t)num_rows,
                         (uint16_t)DIM_C,
                         (uint16_t)DIM_E);
            eu_redmule_wait(&eu_ctrl, WAIT_MODE);

            // Write R3 slice back to L2 at correct offset
            idma_memcpy_1d(&idma_ctrl,
                           1,
                           (uint32_t)r3_out + start_row * DIM_E * 2,
                           obi_r3,
                           num_rows * DIM_E * 2);
            eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
        }
    }

    // Global barrier: wait for Phase 2 to complete
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /**
     * Phase 3: GEMM4 (row-parallel)
     *   GEMM4 group: O = R3 @ M5
     */
    int gemm4_idx = get_local_idx(hartid, gemm4_tiles, GEMM4_N_TILES);
    if (gemm4_idx >= 0) {
        uint32_t start_row, num_rows;
        get_row_range(gemm4_idx, GEMM4_N_TILES, DIM_A, &start_row, &num_rows);

        if (num_rows > 0) {
            // L1 layout: R3_slice, M5, O_slice
            uint32_t obi_r3 = l1_tile_base;
            uint32_t obi_m5 = obi_r3 + (num_rows * DIM_E * 2);
            uint32_t obi_o  = obi_m5 + (DIM_E * DIM_F * 2);

            // Load slice of R3 [num_rows x E] from L2
            idma_memcpy_1d(&idma_ctrl,
                           0,
                           (uint32_t)r3_out + start_row * DIM_E * 2,
                           obi_r3,
                           num_rows * DIM_E * 2);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

            // Load full M5 [ExF] from L2
            idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m5_inp, obi_m5, DIM_E * DIM_F * 2);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

            // Zero accumulator and compute: O_slice = R3_slice @ M5
            mem_set_zero(obi_o, num_rows * DIM_F);
            redmule_gemm(&redmule_ctrl,
                         obi_r3,
                         obi_m5,
                         obi_o,
                         (uint16_t)num_rows,
                         (uint16_t)DIM_E,
                         (uint16_t)DIM_F);
            eu_redmule_wait(&eu_ctrl, WAIT_MODE);

            // Write O slice back to L2 at correct offset
            idma_memcpy_1d(&idma_ctrl,
                           1,
                           (uint32_t)o_out + start_row * DIM_F * 2,
                           obi_o,
                           num_rows * DIM_F * 2);
            eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
        }
    }

    // Global barrier: wait for Phase 3 to complete
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /**
     * Validation: Tile 0 checks O against golden
     */
    uint32_t errors = 0;

    if (hartid == 0) {
        for (uint32_t i = 0; i < DIM_A; i++) {
            for (uint32_t j = 0; j < DIM_F; j++) {
                float16 computed = *(volatile float16 *)(&o_out[i * DIM_F + j]);
                float16 expected = o_golden[i * DIM_F + j];

                uint16_t uc = *(uint16_t *)&computed;
                uint16_t ue = *(uint16_t *)&expected;

                int32_t vc = fp16_to_millis(uc);
                int32_t ve = fp16_to_millis(ue);

                int32_t abs_diff = vc - ve;
                if (abs_diff < 0)
                    abs_diff = -abs_diff;
                if (abs_diff > abs_threshold_millis) {
                    errors++;

                    printf("O[%d][%d]: got=%f (0x%x) exp=%f (0x%x) (abs_diff=%ld)\n",
                           i,
                           j,
                           fp16_to_f64(uc),
                           uc,
                           fp16_to_f64(ue),
                           ue,
                           (long)abs_diff);
                }
            }
        }

        printf("\nTest complete. Errors: %d / %d\n\n", errors, DIM_A * DIM_F);
    }

    return errors;
}
