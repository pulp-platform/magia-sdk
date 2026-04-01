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

int mem_set_zero(uint32_t o, uint32_t dim)
{
    for (uint32_t i = 0; i < dim; i++)
        mmio16(o + i * 2) = 0x0000;
}

/**
 * GEMM chain test with task-level parallelism across tiles.
 *
 * Phase 1 (parallel):
 *   Tile 0: R1[AxC] = M1[AxB] @ M2[BxC]    (GEMM1)
 *   Tile 1: R2[CxE] = M3[CxD] @ M4[DxE]    (GEMM2)
 *
 * Phase 2:
 *   Tile 2: R3[AxE] = R1[AxC] @ R2[CxE]    (GEMM3)
 *
 * Phase 3:
 *   Tile 3: O[AxF]  = R3[AxE] @ M5[ExF]    (GEMM4)
 *
 * Requires minimum 2x2 mesh (4 tiles).
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

    /**
     * Phase 1: GEMM1 and GEMM2 in parallel
     *   Tile 0: R1 = M1 @ M2
     *   Tile 1: R2 = M3 @ M4
     */
    if (hartid == 0) {
        // L1 layout for tile 0: M1, M2, R1
        uint32_t obi_m1 = l1_tile_base;
        uint32_t obi_m2 = obi_m1 + (DIM_A * DIM_B * 2);
        uint32_t obi_r1 = obi_m2 + (DIM_B * DIM_C * 2);

        // Load M1 [AxB] from L2 to L1
        idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m1_inp, obi_m1, DIM_A * DIM_B * 2);
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

        // Load M2 [BxC] from L2 to L1
        idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m2_inp, obi_m2, DIM_B * DIM_C * 2);
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

        // Zero accumulator and compute GEMM1: R1 = M1 @ M2
        mem_set_zero(obi_r1, DIM_A * DIM_C);
        redmule_gemm(&redmule_ctrl,
                     obi_m1,
                     obi_m2,
                     obi_r1,
                     (uint16_t)DIM_A,
                     (uint16_t)DIM_B,
                     (uint16_t)DIM_C);
        eu_redmule_wait(&eu_ctrl, WAIT_MODE);

        // Write R1 [AxC] back to L2
        idma_memcpy_1d(&idma_ctrl, 1, (uint32_t)r1_out, obi_r1, DIM_A * DIM_C * 2);
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    }

    if (hartid == 1) {
        // L1 layout for tile 1: M3, M4, R2
        uint32_t obi_m3 = l1_tile_base;
        uint32_t obi_m4 = obi_m3 + (DIM_C * DIM_D * 2);
        uint32_t obi_r2 = obi_m4 + (DIM_D * DIM_E * 2);

        // Load M3 [CxD] from L2 to L1
        idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m3_inp, obi_m3, DIM_C * DIM_D * 2);
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

        // Load M4 [DxE] from L2 to L1
        idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m4_inp, obi_m4, DIM_D * DIM_E * 2);
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

        // Zero accumulator and compute GEMM2: R2 = M3 @ M4
        mem_set_zero(obi_r2, DIM_C * DIM_E);
        redmule_gemm(&redmule_ctrl,
                     obi_m3,
                     obi_m4,
                     obi_r2,
                     (uint16_t)DIM_C,
                     (uint16_t)DIM_D,
                     (uint16_t)DIM_E);
        eu_redmule_wait(&eu_ctrl, WAIT_MODE);

        // Write R2 [CxE] back to L2
        idma_memcpy_1d(&idma_ctrl, 1, (uint32_t)r2_out, obi_r2, DIM_C * DIM_E * 2);
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    }

    // Global barrier: wait for Phase 1 to complete
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /**
     * Phase 2: GEMM3
     *   Tile 2: R3 = R1 @ R2
     */
    if (hartid == 2) {
        // L1 layout for tile 2: R1, R2, R3
        uint32_t obi_r1 = l1_tile_base;
        uint32_t obi_r2 = obi_r1 + (DIM_A * DIM_C * 2);
        uint32_t obi_r3 = obi_r2 + (DIM_C * DIM_E * 2);

        // Load R1 [AxC] from L2 to L1 (computed by tile 0 in Phase 1)
        idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)r1_out, obi_r1, DIM_A * DIM_C * 2);
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

        // Load R2 [CxE] from L2 to L1 (computed by tile 1 in Phase 1)
        idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)r2_out, obi_r2, DIM_C * DIM_E * 2);
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

        // Zero accumulator and compute GEMM3: R3 = R1 @ R2
        mem_set_zero(obi_r3, DIM_A * DIM_E);
        redmule_gemm(&redmule_ctrl,
                     obi_r1,
                     obi_r2,
                     obi_r3,
                     (uint16_t)DIM_A,
                     (uint16_t)DIM_C,
                     (uint16_t)DIM_E);
        eu_redmule_wait(&eu_ctrl, WAIT_MODE);

        // Write R3 [AxE] back to L2
        idma_memcpy_1d(&idma_ctrl, 1, (uint32_t)r3_out, obi_r3, DIM_A * DIM_E * 2);
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    }

    // Global barrier: wait for Phase 2 to complete
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /**
     * Phase 3: GEMM4
     *   Tile 3: O = R3 @ M5
     */
    if (hartid == 3) {
        // L1 layout for tile 3: R3, M5, O
        uint32_t obi_r3 = l1_tile_base;
        uint32_t obi_m5 = obi_r3 + (DIM_A * DIM_E * 2);
        uint32_t obi_o  = obi_m5 + (DIM_E * DIM_F * 2);

        // Load R3 [AxE] from L2 to L1 (computed by tile 2 in Phase 2)
        idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)r3_out, obi_r3, DIM_A * DIM_E * 2);
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

        // Load M5 [ExF] from L2 to L1
        idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m5_inp, obi_m5, DIM_E * DIM_F * 2);
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

        // Zero accumulator and compute GEMM4: O = R3 @ M5
        mem_set_zero(obi_o, DIM_A * DIM_F);
        redmule_gemm(&redmule_ctrl,
                     obi_r3,
                     obi_m5,
                     obi_o,
                     (uint16_t)DIM_A,
                     (uint16_t)DIM_E,
                     (uint16_t)DIM_F);
        eu_redmule_wait(&eu_ctrl, WAIT_MODE);

        // Write O [AxF] back to L2
        idma_memcpy_1d(&idma_ctrl, 1, (uint32_t)o_out, obi_o, DIM_A * DIM_F * 2);
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
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
                int32_t ic = (uc & 0x8000) ? -(int32_t)(uc & 0x7FFF) : (int32_t)uc;
                int32_t ie = (ue & 0x8000) ? -(int32_t)(ue & 0x7FFF) : (int32_t)ue;
                int32_t ulp_diff = ic - ie;
                if (ulp_diff < 0)
                    ulp_diff = -ulp_diff;
                if (ulp_diff > 17) {
                    errors++;

                    printf("O[%d][%d]: got=%f (0x%x) exp=%f (0x%x)\n",
                           i,
                           j,
                           fp16_to_f64(uc),
                           uc,
                           fp16_to_f64(ue),
                           ue);
                }
            }
        }

        printf("\nTest complete. Errors: %d / %d\n\n", errors, DIM_A * DIM_F);
    }

    return errors;
}
