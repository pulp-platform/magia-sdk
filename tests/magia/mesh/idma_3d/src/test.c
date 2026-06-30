// Copyright 2026 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Calin Diaconu <calin.diaconu@studio.unibo.it>

#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"

#define WAIT_MODE WFE

// Round-trip destination for the main scenario; lives in L2, zero-initialised by default.
static uint8_t dst_3d[D_DEPTH * M_ROWS * N_COLS];

// Edge-case destinations (tile 0 only, also in L2, zero-initialised).
static uint8_t ec_dst_a[M_ROWS * N_COLS];  // edge A: reps3=1
static uint8_t ec_dst_b[D_DEPTH * N_COLS]; // edge B: reps2=1
static uint8_t ec_contiguous_src[D_DEPTH * 2 * 2] = {
    // edge C: no striding
    0x11,
    0x22,
    0x33,
    0x44,
    0x55,
    0x66,
    0x77,
    0x88,
    0x99,
    0xAA,
    0xBB,
    0xCC,
    0xDD,
    0xEE,
    0xFF,
    0x00,
};
static uint8_t ec_dst_c[D_DEPTH * 2 * 2]; // edge C: destination

/**
 * Exercises idma_memcpy_3d in four scenarios:
 *
 * Main  – Every tile copies its [D_DEPTH x tile_h x tile_w] sub-block from
 *         src_3d (strided in L2) to L1, then writes it back to dst_3d.
 *         Tile 0 verifies dst_3d == src_3d across the whole tensor.
 *
 * Edge A – reps3=1 (degenerates to a 2D transfer).
 * Edge B – reps2=1 (one row per plane — tests the outer loop with minimal
 *           inner iterations).
 * Edge C – std2=len, std3=reps2*len (fully contiguous source — no gaps
 *           between rows or planes; all strides equal their dense values).
 *
 * Edge cases A/B/C are run by tile 0 only, after the global barrier.
 */
int main(void)
{
    uint32_t hartid = get_hartid();

    idma_config_t idma_cfg      = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg  = &idma_cfg,
        .api  = &idma_api,
    };
    idma_init(&idma_ctrl);

    fsync_config_t fsync_cfg      = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };
    fsync_init(&fsync_ctrl);

    uint32_t y_id    = GET_Y_ID(hartid);
    uint32_t x_id    = GET_X_ID(hartid);
    uint32_t l1_base = get_l1_base(hartid);

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
#endif

    // -----------------------------------------------------------------------
    // Main scenario: strided 3D sub-block copy
    // -----------------------------------------------------------------------
    uint32_t tile_h_max = (M_ROWS + MESH_Y_TILES - 1) / MESH_Y_TILES;
    uint32_t tile_w_max = (N_COLS + MESH_X_TILES - 1) / MESH_X_TILES;

    int32_t tile_h = ((tile_h_max * y_id) + tile_h_max > M_ROWS)
                         ? (int32_t)(M_ROWS - tile_h_max * y_id)
                         : (int32_t)tile_h_max;
    int32_t tile_w = ((tile_w_max * x_id) + tile_w_max > N_COLS)
                         ? (int32_t)(N_COLS - tile_w_max * x_id)
                         : (int32_t)tile_w_max;

    if (tile_h >= 1 && tile_w >= 1) {
        uint32_t len   = (uint32_t)tile_w; // bytes per innermost row
        uint32_t std2  = N_COLS;           // L2 row stride (bytes)
        uint32_t reps2 = (uint32_t)tile_h; // rows per plane
        uint32_t std3  = M_ROWS * N_COLS;  // L2 plane stride (bytes)
        uint32_t reps3 = D_DEPTH;          // number of planes

        // Corner of this tile's sub-block in the flat L2 arrays
        uint32_t l2_offset = y_id * tile_h_max * N_COLS + x_id * tile_w_max;

        uint32_t axi_src = (uint32_t)src_3d + l2_offset;
        uint32_t axi_dst = (uint32_t)dst_3d + l2_offset;

        // IN: L2 → L1
        idma_memcpy_3d(&idma_ctrl, 0, axi_src, l1_base, len, std2, reps2, std3, reps3);
#if STALLING == 0
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
#endif

        // OUT: L1 → L2
        idma_memcpy_3d(&idma_ctrl, 1, axi_dst, l1_base, len, std2, reps2, std3, reps3);
#if STALLING == 0
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
#endif
    }

    // All tiles rendezvous before verification and edge cases
    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    uint32_t errors = 0;

    if (hartid == 0) {
        // --- Verify main scenario ---
        for (uint32_t i = 0; i < D_DEPTH * M_ROWS * N_COLS; i++) {
            if (dst_3d[i] != src_3d[i])
                errors++;
        }

        // -------------------------------------------------------------------
        // Edge A: reps3=1  →  outer-plane loop fires exactly once.
        // Transfers the full plane-0 block (all rows, all cols) of src_3d.
        // std2==len here (rows in plane 0 are contiguous in src_3d) so this
        // also exercises the case where the 2D stride equals the row length.
        // -------------------------------------------------------------------
        {
            uint32_t len_a   = N_COLS;
            uint32_t std2_a  = N_COLS; // == len_a: contiguous rows
            uint32_t reps2_a = M_ROWS;
            uint32_t std3_a  = M_ROWS * N_COLS; // irrelevant: reps3=1
            uint32_t reps3_a = 1;

            idma_memcpy_3d(
                &idma_ctrl, 0, (uint32_t)src_3d, l1_base, len_a, std2_a, reps2_a, std3_a, reps3_a);
#if STALLING == 0
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
#endif

            idma_memcpy_3d(&idma_ctrl,
                           1,
                           (uint32_t)ec_dst_a,
                           l1_base,
                           len_a,
                           std2_a,
                           reps2_a,
                           std3_a,
                           reps3_a);
#if STALLING == 0
            eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
#endif

            // ec_dst_a[i] must equal plane-0 of src_3d (first M_ROWS*N_COLS bytes)
            for (uint32_t i = 0; i < M_ROWS * N_COLS; i++) {
                if (ec_dst_a[i] != src_3d[i])
                    errors++;
            }
        }

        // -------------------------------------------------------------------
        // Edge B: reps2=1  →  inner loop fires exactly once per plane.
        // Copies only the first row of each of the D_DEPTH planes.
        // Tests that the outer (plane) loop works correctly even when the
        // middle loop contributes a single iteration.
        // -------------------------------------------------------------------
        {
            uint32_t len_b   = N_COLS;
            uint32_t std2_b  = N_COLS; // irrelevant: reps2=1
            uint32_t reps2_b = 1;
            uint32_t std3_b  = M_ROWS * N_COLS; // skip to next plane's row-0
            uint32_t reps3_b = D_DEPTH;

            // IN: first row of each plane → contiguous in L1
            // L1 layout: plane d occupies L1[d*N_COLS .. d*N_COLS+N_COLS-1]
            idma_memcpy_3d(
                &idma_ctrl, 0, (uint32_t)src_3d, l1_base, len_b, std2_b, reps2_b, std3_b, reps3_b);
#if STALLING == 0
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
#endif

            // OUT: L1 → ec_dst_b (also contiguous, std3=N_COLS)
            idma_memcpy_3d(&idma_ctrl,
                           1,
                           (uint32_t)ec_dst_b,
                           l1_base,
                           len_b,
                           std2_b,
                           reps2_b,
                           N_COLS, // plane stride in ec_dst_b
                           reps3_b);
#if STALLING == 0
            eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
#endif

            // ec_dst_b[d*N_COLS + c] == src_3d[d*M_ROWS*N_COLS + c]
            for (uint32_t d = 0; d < D_DEPTH; d++) {
                for (uint32_t c = 0; c < N_COLS; c++) {
                    if (ec_dst_b[d * N_COLS + c] != src_3d[d * M_ROWS * N_COLS + c])
                        errors++;
                }
            }
        }

        // -------------------------------------------------------------------
        // Edge C: fully contiguous source (std2==len, std3==reps2*len).
        // No gaps between rows or planes; the 3D engine is exercised with
        // parameters that make the three loops equivalent to one flat copy.
        // -------------------------------------------------------------------
        {
            uint32_t len_c   = 2;
            uint32_t reps2_c = 2;
            uint32_t reps3_c = D_DEPTH;
            uint32_t std2_c  = len_c;           // contiguous rows
            uint32_t std3_c  = reps2_c * len_c; // contiguous planes

            idma_memcpy_3d(&idma_ctrl,
                           0,
                           (uint32_t)ec_contiguous_src,
                           l1_base,
                           len_c,
                           std2_c,
                           reps2_c,
                           std3_c,
                           reps3_c);
#if STALLING == 0
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
#endif

            idma_memcpy_3d(&idma_ctrl,
                           1,
                           (uint32_t)ec_dst_c,
                           l1_base,
                           len_c,
                           std2_c,
                           reps2_c,
                           std3_c,
                           reps3_c);
#if STALLING == 0
            eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
#endif

            for (uint32_t i = 0; i < D_DEPTH * 2 * 2; i++) {
                if (ec_dst_c[i] != ec_contiguous_src[i])
                    errors++;
            }
        }
    }

    printf("Number of errors: %d\n", errors);
    return errors;
}
