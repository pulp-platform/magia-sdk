// Copyright 2025 University of Bologna
// Licensed under the Apache License, Version 2.0
// SPDX-License-Identifier: Apache-2.0
//
// Measures cycles/byte for streaming transfering data from L2 to L1 and viceversa.
// Tests both using the DMA, and not using the DMA.
// Only one tile at a time is doing the data transfer.

#include <stdint.h>

#include "test.h"
#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"

#define WAIT_MODE                           WFE

#define N_BUFFS                             1 // Number of Kilobytes to be transferred
#define BUF_SIZE                            (N_BUFFS * 1024) // Total data buffer size
#define REPEATS                             1                // multiple repeats for averaging

#define L2_SRC_BASE                         0xCC040000 // Arbitrary L2 base address for this test

// Theoretical peak bandwidth of the NoC (bytes per cycle)
// For a 256-bit wide bus: 32 bytes/cycle
#define THEORETICAL_PEAK_BW_BYTES_PER_CYCLE 32

int main(void)
{
    uint32_t hartid       = get_hartid();
    uint32_t l1_tile_base = get_l1_base(hartid);

    // Init DMA
    idma_config_t idma_cfg      = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg  = &idma_cfg,
        .api  = &idma_api,
    };
    idma_init(&idma_ctrl);

    // Init FSYNC
    fsync_config_t fsync_cfg      = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };
    fsync_init(&fsync_ctrl);

// Init EU
#if STALLING == 0
    eu_config_t eu_cfg      = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg  = &eu_cfg,
        .api  = &eu_api,
    };
    eu_init(&eu_ctrl);
    eu_idma_init(&eu_ctrl, 0);
    eu_fsync_init(&eu_ctrl, 0);
#endif

    // Allocate private L2 buffers for this tile
    uint8_t *src_buf;
    uint8_t *dst_buf;

    src_buf = (uint8_t *)(L2_SRC_BASE + hartid * BUF_SIZE * 2);
    dst_buf = (uint8_t *)(L2_SRC_BASE + hartid * BUF_SIZE * 2 + BUF_SIZE);

    // Sync all tiles before measurement
    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    // for(uint32_t i = 0; i < NUM_HARTS; i++){
    // if(hartid == i){
    // --- L2 -> L1 --- WITH IDMA
    uint32_t start_l2l1 = perf_get_cycles();
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)src_buf, l1_tile_base, BUF_SIZE);
#if STALLING == 0
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
#endif
    uint32_t stop_l2l1 = perf_get_cycles();

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    // --- L1 -> L2 --- WITH IDMA
    uint32_t start_l1l2 = perf_get_cycles();
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t)dst_buf, l1_tile_base, BUF_SIZE);
#if STALLING == 0
    eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
#endif
    uint32_t stop_l1l2 = perf_get_cycles();

    printf("[TILE %d]: IDMA L2 -> L1: %d cycles\n", hartid, (stop_l2l1 - start_l2l1));
    printf("[TILE %d]: IDMA L1 -> L2: %d cycles\n", hartid, (stop_l1l2 - start_l1l2));

    start_l2l1 = perf_get_cycles();
    for (uint32_t j = 0; j < BUF_SIZE; j += 4) {
        *(uint32_t *)(l1_tile_base + j) = mmio32(src_buf + j);
    }
    stop_l2l1 = perf_get_cycles();

    start_l1l2 = perf_get_cycles();
    for (uint32_t j = 0; j < BUF_SIZE; j += 4) {
        *(uint32_t *)(src_buf + j) = mmio32(l1_tile_base + j);
    }
    stop_l1l2 = perf_get_cycles();

    printf("[TILE %d]: NO_IDMA L2 -> L1: %d cycles\n", hartid, (stop_l2l1 - start_l2l1));
    printf("[TILE %d]: NO_IDMA L1 -> L2: %d cycles\n", hartid, (stop_l1l2 - start_l1l2));
    // }
    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif
    // }

    return 0;
}