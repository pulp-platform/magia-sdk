// Copyright 2025 University of Bologna
// Licensed under the Apache License, Version 2.0
// SPDX-License-Identifier: Apache-2.0
//
// Multi-tile memory-bound benchmark for MAGIA
// Measures cycles/byte for streaming DMA transfers + memory accesses.
// Now with separate L2->L1 and L1->L2 timing, bandwidth, and utilisation.

#include <stdint.h>

#include "test.h"
#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"

#define WAIT_MODE WFE

#define BUF_SIZE   (16 * 1024)        // 16 KB buffer (matches reference)
#define REPEATS    1                 // multiple repeats for averaging
#define DMA_CHUNK_SIZE 0x4000         // 16 KB per DMA chunk (unused but kept)

#define L2_SRC_BASE  0xCC040000       // L2 base address for this test

// Theoretical peak bandwidth of the NoC (bytes per cycle)
// For a 256-bit wide bus: 32 bytes/cycle
#define THEORETICAL_PEAK_BW_BYTES_PER_CYCLE 32

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

    // Allocate private L2 buffers for this tile
    uint8_t* src_buf; 
    uint8_t* dst_buf;

    src_buf = (uint8_t*)(L2_SRC_BASE + hartid * BUF_SIZE * 2);
    dst_buf = (uint8_t*)(L2_SRC_BASE + hartid * BUF_SIZE * 2 + BUF_SIZE);

    // Fill source buffer with a deterministic pattern
    // for (int i = 0; i < BUF_SIZE; i += 4) {
    //     *(uint32_t*)(src_buf + i) = (uint32_t)(i & 0xFFFFFFFF);
    //     *(uint32_t*)(dst_buf + i) = 0x00000000;   // clear destination
    // }

    // Sync all tiles before measurement
    fsync_sync_global(&fsync_ctrl);
    #if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    #endif

    // Arrays to store cycles for each repetition
    uint32_t cycles_l2l1[REPEATS];
    uint32_t cycles_l1l2[REPEATS];
    uint32_t cycles_total[REPEATS];

    // Benchmark loop
    for (int r = 0; r < REPEATS; r++) {
        sentinel_start();

        // --- L2 -> L1 ---
        uint32_t start_l2l1 = perf_get_cycles();
        idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)src_buf, l1_tile_base, BUF_SIZE);
        #if STALLING == 0
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
        #endif
        uint32_t stop_l2l1 = perf_get_cycles();

        // --- L1 -> L2 ---
        uint32_t start_l1l2 = perf_get_cycles();
        idma_memcpy_1d(&idma_ctrl, 1, (uint32_t)dst_buf, l1_tile_base, BUF_SIZE);
        #if STALLING == 0
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
        #endif
        uint32_t stop_l1l2 = perf_get_cycles();

        sentinel_end();

        cycles_l2l1[r] = stop_l2l1 - start_l2l1;
        cycles_l1l2[r] = stop_l1l2 - start_l1l2;
        cycles_total[r] = cycles_l2l1[r] + cycles_l1l2[r];
    }

    // Verification: compare source and destination buffers
    uint32_t errors = 0;
    for (int i = 0; i < BUF_SIZE; i++) {
        uint8_t expected = *(volatile uint8_t*)(src_buf + i);
        uint8_t got      = *(volatile uint8_t*)(dst_buf + i);
        if (expected != got) {
            errors++;
        }
    }

    // Compute averages over REPEATS
    uint32_t sum_l2l1 = 0, sum_l1l2 = 0, sum_total = 0;
    for (int r = 0; r < REPEATS; r++) {
        sum_l2l1 += cycles_l2l1[r];
        sum_l1l2 += cycles_l1l2[r];
        sum_total += cycles_total[r];
    }
    uint32_t avg_l2l1 = sum_l2l1 / REPEATS;
    uint32_t avg_l1l2 = sum_l1l2 / REPEATS;
    uint32_t avg_total = sum_total / REPEATS;

    uint32_t total_bytes = 2 * BUF_SIZE;   // total bytes moved per repetition

    // Bandwidth in bytes/cycle scaled by 1000 (fixed?point)
    uint32_t bw_l2l1_x1000 = (BUF_SIZE * 1000) / avg_l2l1;
    uint32_t bw_l1l2_x1000 = (BUF_SIZE * 1000) / avg_l1l2;
    uint32_t bw_total_x1000 = (total_bytes * 1000) / avg_total;

    // Bandwidth utilisation (actual total BW / peak BW) ×1000
    // Actual total BW = total_bytes / avg_total (bytes/cycle)
    // Peak BW = THEORETICAL_PEAK_BW_BYTES_PER_CYCLE
    uint32_t util_x1000 = (total_bytes * 1000) / (avg_total * THEORETICAL_PEAK_BW_BYTES_PER_CYCLE);

    // Sync all tiles before printing (optional, to avoid interleaved output)
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);

    // Print per?tile results (similar to the reference benchmark)
    printf("Tile %u | "
           "L2->L1: %4u cyc (%3u B/c x1000) | "
           "L1->L2: %4u cyc (%3u B/c x1000) | "
           "Total:  %4u cyc (%3u B/c x1000) | "
           "Util: %3u/1000 | errors=%u\n",
           hartid,
           avg_l2l1, bw_l2l1_x1000,
           avg_l1l2, bw_l1l2_x1000,
           avg_total, bw_total_x1000,
           util_x1000, errors);

    return (int)errors;
}