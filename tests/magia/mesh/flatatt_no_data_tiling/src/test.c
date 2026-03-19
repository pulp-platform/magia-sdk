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
 * This test aims to verify the functionality of MAGIA for the FlatAttention algorithm.
 *
 * Operations:
 *      2.1. Q @ K^T -> S
 *      2.2. row_max(S) -> M
 *      2.3. S - M -> S
 *      2.4. exp(S) -> S
 *      2.5. row_sum(S) -> L
 *      2.6. S @ V -> O
 *      2.7. O / L -> O
 */
int main(void)
{
    /**
     * 0. Test initializations
     */

    // Get constants
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
     * 1. Operation initializations
     */

    // Calculate tile_h and tile_w
    uint32_t tile_h_max = ((S_SIZE + MESH_Y_TILES - 1) / MESH_Y_TILES);
    uint32_t tile_w_max = ((S_SIZE + MESH_X_TILES - 1) / MESH_X_TILES);
    int32_t tile_h;
    int32_t tile_w;

    if (((tile_h_max * y_id) + tile_h_max) > S_SIZE)
        tile_h = S_SIZE - (tile_h_max * y_id);
    else
        tile_h = tile_h_max;

    if (((tile_w_max * x_id) + tile_w_max) > S_SIZE)
        tile_w = S_SIZE - (tile_w_max * x_id);
    else
        tile_w = tile_w_max;

    // Initialize the L1 addresses for the tiles
    uint32_t obi_addr_m = l1_tile_base;
    uint32_t obi_addr_l = obi_addr_m + (tile_h * 2);
    uint32_t obi_addr_o = obi_addr_l + (tile_h * 2);
    uint32_t obi_addr_q = obi_addr_o + (tile_h * D_SIZE * 2);
    uint32_t obi_addr_k = obi_addr_q + (tile_h * D_SIZE * 2);
    uint32_t obi_addr_v = obi_addr_k + (tile_w * D_SIZE * 2);
    uint32_t obi_addr_s = obi_addr_v + (tile_w * D_SIZE * 2);

    // Initialize the L2 addresses for the tiles
    uint32_t len_q      = D_SIZE * 2;
    uint32_t std_q      = D_SIZE * 2;
    uint32_t reps_q     = (uint32_t)tile_h;
    uint32_t axi_addr_q = (uint32_t)q_inp + (y_id * tile_h_max * D_SIZE * 2);

    uint32_t len_k      = tile_w * 2;
    uint32_t std_k      = S_SIZE * 2;
    uint32_t reps_k     = (uint32_t)D_SIZE;
    uint32_t axi_addr_k = (uint32_t)k_inp + (x_id * tile_w_max * 2);

    uint32_t len_v      = D_SIZE * 2;
    uint32_t std_v      = D_SIZE * 2;
    uint32_t reps_v     = (uint32_t)tile_w;
    uint32_t axi_addr_v = (uint32_t)v_inp + (x_id * tile_w_max * D_SIZE * 2);

    /**
     * 2. Perform layer operations
     *
     * 2.1. Compute Q @ K^T -> S
     * K is stored in k_inp directly transposed
     */

    // Load the Q and K data-tiles (AXI -> OBI, L2 -> L1)
    idma_memcpy_2d(&idma_ctrl, 0, axi_addr_q, obi_addr_q, len_q, std_q, reps_q);
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    idma_memcpy_2d(&idma_ctrl, 0, axi_addr_k, obi_addr_k, len_k, std_k, reps_k);
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    // Required because the RedMulE GEMM accumulates in Y (obi_addr_s)
    mem_set_zero(obi_addr_s, tile_h * tile_w);

    // Perform actual matmul
    redmule_gemm(&redmule_ctrl,
                 obi_addr_q,
                 obi_addr_k,
                 obi_addr_s,
                 (uint16_t)tile_h,
                 (uint16_t)D_SIZE,
                 (uint16_t)tile_w);
    eu_redmule_wait(&eu_ctrl, WAIT_MODE);

    /**
     * 2.2. Find row maxes. Store in M.
     * row_max(S) -> M
     *
     * Each tile compares with the max of the previous tile, and propagates them
     * to the following tile. Then, propagate the values back to the entire row.
     */
    // Find the tile-max of each row in S, and store it in M
    row_max(obi_addr_s, obi_addr_m, tile_h, tile_w);

    // Propagate the maxes to the right tile, comparing with the maxes of the previous tile
    if (x_id != 0) {
        // Wait for the left tile to finish computing the maxes
        fsync_sync_left(&fsync_ctrl);
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);

        // Compare with the maxes of the left tile and update M if needed
        max_compare(obi_addr_m, get_l1_base(hartid - 1), tile_h);
    }

    if (x_id != (MESH_X_TILES - 1)) {
        // Wait for the right tile to finish comparing with the left tile
        fsync_sync_right(&fsync_ctrl);
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    }

    // Wait for the entire tile row to finish comparing
    fsync_sync_row(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // Propagate the maxes from rightmost tile back to the entire tile row and store in M
    if (x_id != (MESH_X_TILES - 1)) {
        max_compare(obi_addr_m, get_l1_base(GET_ID(y_id, (MESH_X_TILES - 1))), tile_h);
    }

    // Wait for the entire tile row to finish comparing
    fsync_sync_row(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /**
     * 2.3. Element-wise substraction for each row with their maximum (stored in M)
     * S - M -> S
     */
    rowdiff(obi_addr_s, obi_addr_m, tile_h, tile_w);

    /**
     * 2.4. In-place element-wise exponential on the scores (stored in S)
     * exp(S) -> S
     */
    exponential(obi_addr_s, (uint32_t)tile_h, (uint32_t)tile_w);

    /**
     * 2.5. Row-wise summation of the elements. Store in L.
     * row_sum(S) -> L
     *
     * After summing, add the contribution of the previous tile and send them forward.
     * When reaching the end, "broadcast" it back.
     */
    // Tile-wide row summation, stored in L
    row_sum(obi_addr_s, obi_addr_l, tile_h, tile_w);

    // Propagate the accumulated sum from the left tile to the right-most tile:
    // add current tile to the accumulated sum, and propagate to the right tile
    if (x_id != 0) {
        fsync_sync_left(&fsync_ctrl);
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);
        vect_sum(obi_addr_l, get_l1_base(hartid - 1) + (tile_h * 2), tile_h);
    }

    // Propagate the accumulated sum to the right tile
    if (x_id != (MESH_X_TILES - 1)) {
        fsync_sync_right(&fsync_ctrl);
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    }

    // Wait for the entire tile row to finish summing
    fsync_sync_row(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // Propagate the accumulated sum back to the entire tile row and store in L
    if (x_id != (MESH_X_TILES - 1)) {
        idma_memcpy_1d(&idma_ctrl,
                       0,
                       get_l1_base(GET_ID(y_id, (MESH_X_TILES - 1))) + (tile_h * 2),
                       obi_addr_l,
                       tile_h * 2);
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    }

    // Wait for the entire tile row to finish propagating
    fsync_sync_row(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /**
     * 2.6. Compute S @ V -> O
     */
    // Load the V data-tile (AXI -> OBI, L2 -> L1)
    idma_memcpy_2d(&idma_ctrl, 0, axi_addr_v, obi_addr_v, len_v, std_v, reps_v);
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    // Required because the RedMule GEMM accumulates in Y (obi_addr_o)
    mem_set_zero(obi_addr_o, tile_h * D_SIZE);

    // Perform actual matmul
    redmule_gemm(&redmule_ctrl,
                 obi_addr_s,
                 obi_addr_v,
                 obi_addr_o,
                 (uint16_t)tile_h,
                 (uint16_t)tile_w,
                 (uint16_t)D_SIZE);
    eu_redmule_wait(&eu_ctrl, WAIT_MODE);

    /**
     * 2.7. Divide the output buffer by the accumulated sum buffer
     * O / L -> O
     */
    rowdiv(obi_addr_o, obi_addr_l, tile_h, D_SIZE);

    // Propagate the output to the right tile, adding the contribution of the previous tiles
    if (x_id != 0) {
        fsync_sync_left(&fsync_ctrl);
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);

        vect_sum(obi_addr_o, get_l1_base(hartid - 1) + (tile_h * 4), tile_h * D_SIZE);
    }

    // Copy output to L2 if this is the rightmost tile
    if (x_id != (MESH_X_TILES - 1)) {
        fsync_sync_right(&fsync_ctrl);
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    } else {
        idma_memcpy_2d(&idma_ctrl,
                       1,
                       (uint32_t)o_out + (y_id * tile_h_max * D_SIZE * 2),
                       obi_addr_o,
                       D_SIZE * 2,
                       D_SIZE * 2,
                       tile_h);
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    }

    // Wait for the entire tile row to finish propagating
    fsync_sync_row(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // Synchronize all tiles
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /*
     * Check results using tile 0
     */
    uint32_t errors = 0;

    if (hartid == 0) {
        float16alt threshold = (float16alt)0.02f;

        for (uint32_t i = 0; i < S_SIZE; i++) {
            for (uint32_t j = 0; j < D_SIZE; j++) {
                float16alt computed = *(volatile float16alt *)(&o_out[i * D_SIZE + j]);
                float16alt expected = o_golden[i * D_SIZE + j];
                float16alt diff =
                    (computed > expected) ? (computed - expected) : (expected - computed);

                if (diff > threshold) {
                    printf("Error at [%d][%d]: got=%x exp=%x\n",
                           i,
                           j,
                           *(uint16_t *)&computed,
                           *(uint16_t *)&expected);

                    errors++;
                }
            }
        }

        printf("\nTest is over.\nNumber of errors: %d / %d\n\n", errors, S_SIZE * D_SIZE);
    }

    return errors;
}
