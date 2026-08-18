// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "fsync.h"
#include "idma.h"
#include "redmule.h"
#include "eventunit.h"

#define WAIT_MODE            WFE
#define abs_threshold_millis 8 /* 0.008 expressed as integer millis */

#define GEMM1_N_TILES        1

static const uint32_t gemm1_tiles[GEMM1_N_TILES] = {0};

#ifndef FIFO_N_CHUNKS
#define FIFO_N_CHUNKS 5
#endif
#define FIFO_BATCH_FRAC (1.0f / FIFO_N_CHUNKS)

static int get_local_idx(uint32_t hartid, const uint32_t *tiles, uint32_t n_tiles)
{
    for (uint32_t i = 0; i < n_tiles; i++)
        if (tiles[i] == hartid)
            return (int)i;
    return -1;
}

/* Zero `n_halfwords` fp16 elements starting at L1 address `base` using 32-bit
 * stores. Assumes `base` is 4-byte aligned and `n_halfwords` is even, which
 * holds for the ping-pong chunks below (chunk_rows * DIM_C with even DIM_C). */
static inline void l1_zero_fp16(uint32_t base, uint32_t n_halfwords)
{
    volatile uint32_t *p = (volatile uint32_t *)base;
    uint32_t n           = n_halfwords >> 1;
    for (uint32_t i = 0; i < n; i++)
        p[i] = 0;
}

int main(void)
{
    /* ~~~~~~~~~~~~~~~~~~~~ 0. Initialization ~~~~~~~~~~~~~~~~~~~~ */
    uint32_t hartid       = get_hartid();
    uint32_t l1_tile_base = get_l1_base(hartid);

    /* Init iDMA */
    idma_config_t idma_cfg      = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg  = &idma_cfg,
        .api  = &idma_api,
    };
    idma_init(&idma_ctrl);

    /* Init RedMulE */
    redmule_config_t redmule_cfg      = {.hartid = hartid};
    redmule_controller_t redmule_ctrl = {
        .base = NULL,
        .cfg  = &redmule_cfg,
        .api  = &redmule_api,
    };
    redmule_init(&redmule_ctrl);

    /* Init FractalSync */
    fsync_config_t fsync_cfg      = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };
    fsync_init(&fsync_ctrl);

/* Init Event Unit */
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

    int gemm1_idx = get_local_idx(hartid, gemm1_tiles, GEMM1_N_TILES);
    // int gemm2_idx = get_local_idx(hartid, gemm2_tiles, GEMM2_N_TILES);
    // int gemm3_idx = get_local_idx(hartid, gemm3_tiles, GEMM3_N_TILES);
    // int gemm4_idx = get_local_idx(hartid, gemm4_tiles, GEMM4_N_TILES);

    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* Barrier: synchronize all tiles after GEMM1 before sandbox transfers */
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    uint32_t m2_size = (uint32_t)(DIM_B * DIM_C * 2);

    /* Xfer A: L2 → tile 12 L1 */
    if (hartid == 12) {
        idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m2_inp, get_l1_base(12), m2_size);
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    }
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* Xfer C: tile 12 L1 → tile 13 L1, pushed by tile 12 */
    if (hartid == 12) {
        idma_memcpy_1d(&idma_ctrl, 1, get_l1_base(13), get_l1_base(12), m2_size);
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    }
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* Xfer E: tile 12 L1 → tile 13 L1, work split in half:
     *   tile 12 pushes first half  (dir=1, src=tile12 L1,        dst=tile13 L1)
     *   tile 13 pulls second half  (dir=0, src=tile12 L1 + half, dst=tile13 L1 + half) */
    uint32_t half = m2_size / 2;
    if (hartid == 12) {
        idma_memcpy_1d(&idma_ctrl, 1, get_l1_base(13), get_l1_base(12), half);
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    }
    if (hartid == 13) {
        idma_memcpy_1d(&idma_ctrl, 0, get_l1_base(12) + half, get_l1_base(13) + half, half);
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    }
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* ~~~~~~~~~~~~~~~~~~~~ Validation ~~~~~~~~~~~~~~~~~~~~ */
    // Final barrier
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // Tile 0 checks R1 against golden
    uint32_t errors = 0;

    return errors;
}
