// Copyright 2025 University of Bologna
// Licensed under the Apache License, Version 2.0
// SPDX-License-Identifier: Apache-2.0
//
// Multi-tile memory-bound benchmark for MAGIA
// Measures cycles/byte for streaming DMA transfers + memory accesses.

#include <stdint.h>
#include <stdio.h>

#include "test.h"
#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"

#define WAIT_MODE WFE

#define BUF_SIZE   (16 * 1024)   // 16KB buffer
#define REPEATS    1             // reduced from 64 to avoid counter issues
#define DMA_CHUNK_SIZE 0x4000    // 16 KB per DMA chunk (safe)

#define L2_SRC_BASE     0xCC040000

int main(void) {
    uint32_t hartid = get_hartid();
    uint32_t l1_tile_base = get_l1_base(hartid);

    // Init DMA
    idma_config_t idma_cfg = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg  = &idma_cfg,
        .api  = &idma_api,
    };
    idma_init(&idma_ctrl);

    // Init FSYNC
    fsync_config_t fsync_cfg = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };
    fsync_init(&fsync_ctrl);

    #if STALLING == 0
    eu_config_t eu_cfg = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg = &eu_cfg,
        .api = &eu_api,
    };
    eu_init(&eu_ctrl);
    eu_idma_init(&eu_ctrl, 0);
    eu_fsync_init(&eu_ctrl, 0);
    #endif

    uint8_t* src_buf; 
    uint8_t* dst_buf;

    src_buf = (L2_SRC_BASE + hartid * BUF_SIZE * 2);
    dst_buf = (L2_SRC_BASE + hartid * BUF_SIZE * 2 + BUF_SIZE);

    // Fill source buffer
    for (int i = 0; i < BUF_SIZE; i += 4) {
        *(uint32_t*) (src_buf + i) = (uint32_t)(i & 0xFFFFFFFF);
        *(uint32_t*) (dst_buf + i) = (uint32_t)(0x00000000);
    }

    // Sync all tiles before measurement
    fsync_sync_global(&fsync_ctrl);
    #if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    #endif

    // Array to store cycles for each repetition
    uint32_t cycles_array[REPEATS];

    // perf_reset();
    // perf_start();

    // Repeat DMA copies to stress memory, measure per repetition
    for (int r = 0; r < REPEATS; r++) {
        // uint32_t cycle_start = perf_get_cycles();
        sentinel_start();
        
        uint32_t src_addr = (uint32_t)src_buf;
        uint32_t dst_addr = (uint32_t)l1_tile_base;

        idma_memcpy_1d(&idma_ctrl, 0, src_addr, dst_addr, (uint32_t) BUF_SIZE);
        #if STALLING == 0
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
        #endif


        src_addr = (uint32_t)l1_tile_base;
        dst_addr = (uint32_t)dst_buf;
        idma_memcpy_1d(&idma_ctrl, 1, dst_addr, src_addr, (uint32_t) BUF_SIZE);
        #if STALLING == 0
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
        #endif

        sentinel_end();

        // uint32_t cycle_stop = perf_get_cycles();
        // *(uint32_t*)(cycles_array + r) = (uint32_t) (cycle_stop - cycle_start);
        // perf_reset();
    }


    uint8_t counter = 0;
    for (int i = 0; i < BUF_SIZE; i++) {
        uint8_t computed  = *(volatile uint8_t*)(src_buf + (i));
        uint8_t expected = *(volatile uint8_t*)(dst_buf + (i));
        if(computed!=expected){
            #if EVAL == 1
            printf("Giuda faus t %d %d\n", expected, computed);
            #endif
        }
        counter++;
        if(counter == 10)
            return 0;   
    }

    // Compute average cycles
    // uint32_t total_cycles = 0;
    // for (int r = 0; r < REPEATS; r++) {
    //     total_cycles += *(volatile uint32_t*)(cycles_array + r);
    // }
    // uint32_t avg_cycles = total_cycles / REPEATS;

    // // Total bytes moved per repetition = 2 * BUF_SIZE
    // uint32_t total_bytes = 2 * BUF_SIZE;

    // // Calculate bytes/cycle as scaled integer (x1000 to keep 3 decimals)
    // uint32_t perf_scaled = (total_bytes) / avg_cycles;

    // // Sync all tiles before printing
    // fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);

    // // Each tile prints its own performance
    // printf("Tile %u | Total bytes per repetition: %u | Avg cycles: %u | Performance: %u bytes/cycle\n",
    //        hartid, total_bytes, avg_cycles, perf_scaled);

    return 0;
}