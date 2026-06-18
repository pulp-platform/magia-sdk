// Copyright 2025 University of Bologna
// Licensed under the Apache License, Version 2.0
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: Statistical repeatability — variance and jitter analysis
//
// Purpose:
//   All prior benchmarks report a single average.  This benchmark quantifies
//   measurement variance to establish whether cycle-count results are stable
//   enough to support claims about small performance differences.
//
//   For each of three representative transfer types:
//     (A) L2→L1 contiguous (same as memory-bound benchmark)
//     (B) Tile-to-tile 1-hop ring shift (same as ring-shift benchmark)
//     (C) Global fsync barrier
//
//   It runs TRIAL_COUNT independent measurements and computes:
//     - Mean
//     - Min and Max
//     - Range (max - min)
//     - Coefficient of Variation (CV) = (range / mean) * 1000  (per-mille)
//
//   A CV < 10 per-mille (1%) indicates that single-run results are reliable.
//   A CV > 50 per-mille suggests multi-run averaging is mandatory.
//
//   This benchmark should be run first when bringing up a new mesh size to
//   determine the minimum REPEATS needed in other benchmarks.

#include <stdint.h>

#include "test.h"
#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"

#define WAIT_MODE    WFE
#define TRIAL_COUNT  64    // independent back-to-back measurements per case

// Buffer sizes matching the reference benchmarks for direct comparison
#define BUF_SIZE_L2   (16 * 1024)   // 16 KB, same as memory-bound bench
#define BUF_SIZE_NOC  64            // 64 B, same as ring-shift bench

// L2 source for case A
#define L2_BASE         0xCC040000UL
#define L2_TILE_STRIDE  0x00008000UL   // 32 KB per tile (>= BUF_SIZE_L2)

// L1 layout
#define L1_SRC_OFFSET  0x00021000
#define L1_DST_OFFSET  0x00022000
#define L1_TMP_OFFSET  0x00023000   // scratch for case B incoming data

// Per-tile per-case statistics published for tile-0 summary
typedef struct {
    uint32_t mean;
    uint32_t min;
    uint32_t max;
    uint32_t cv_permille;   // (range / mean) * 1000
} stat_t;

volatile stat_t tile_stat_a[MESH_X_TILES * MESH_Y_TILES];
volatile stat_t tile_stat_b[MESH_X_TILES * MESH_Y_TILES];
volatile stat_t tile_stat_c[MESH_X_TILES * MESH_Y_TILES];

// Compute statistics from an array of cycle measurements
static void compute_stats(const uint32_t *samples, uint32_t n, stat_t *out)
{
    uint32_t sum = 0;
    uint32_t mn  = samples[0];
    uint32_t mx  = samples[0];

    for (uint32_t i = 0; i < n; i++) {
        sum += samples[i];
        if (samples[i] < mn) mn = samples[i];
        if (samples[i] > mx) mx = samples[i];
    }

    out->mean = sum / n;
    out->min  = mn;
    out->max  = mx;
    // CV = (range / mean) * 1000, expressed as per-mille integer
    out->cv_permille = (out->mean > 0) ?
                       ((mx - mn) * 1000u) / out->mean : 0;
}

int main(void)
{
    uint32_t hartid    = get_hartid();
    uint32_t x_id      = GET_X_ID(hartid);
    uint32_t y_id      = GET_Y_ID(hartid);
    uint32_t tile_base = hartid * 0x00100000;

    uint8_t  *l1_src = (uint8_t *)(tile_base + L1_SRC_OFFSET);
    uint8_t  *l1_dst = (uint8_t *)(tile_base + L1_DST_OFFSET);
    uint8_t  *l1_tmp = (uint8_t *)(tile_base + L1_TMP_OFFSET);

    uint32_t l2_src = (uint32_t)(L2_BASE + (uint32_t)hartid * L2_TILE_STRIDE);

    // Sample arrays stored locally (TRIAL_COUNT * 4 bytes each, in registers/stack)
    uint32_t samples_a[TRIAL_COUNT];
    uint32_t samples_b[TRIAL_COUNT];
    uint32_t samples_c[TRIAL_COUNT];

    // ----------------------------------------------------------------
    // Controller initialisation
    // ----------------------------------------------------------------

    idma_config_t idma_cfg = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg  = &idma_cfg,
        .api  = &idma_api,
    };
    idma_init(&idma_ctrl);

    fsync_config_t fsync_cfg = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };
    fsync_init(&fsync_ctrl);

    eu_config_t eu_cfg = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg  = &eu_cfg,
        .api  = &eu_api,
    };
    eu_init(&eu_ctrl);
    eu_idma_init(&eu_ctrl, 0);
    eu_fsync_init(&eu_ctrl, 0);

    // ----------------------------------------------------------------
    // Buffer initialisation
    // ----------------------------------------------------------------

    for (int i = 0; i < BUF_SIZE_L2; i += 4)
        *(volatile uint32_t *)(l2_src + i) = (uint32_t)(hartid * 0x1000 + i);

    for (int i = 0; i < BUF_SIZE_NOC; i++) {
        l1_src[i] = (uint8_t)(hartid + i);
        l1_dst[i] = 0;
        l1_tmp[i] = 0;
    }

    // Warm-up global barrier
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // ================================================================
    // CASE A: L2 → L1 contiguous transfer (16 KB)
    // ================================================================

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    for (int t = 0; t < TRIAL_COUNT; t++) {
        uint32_t t0 = perf_get_cycles();

        idma_memcpy_1d(&idma_ctrl,
                       0,          // AXI→OBI
                       l2_src,
                       (uint32_t)l1_dst,
                       BUF_SIZE_L2);

        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

        uint32_t t1 = perf_get_cycles();
        samples_a[t] = t1 - t0;
    }

    {
        stat_t s;
        compute_stats(samples_a, TRIAL_COUNT, &s);
        tile_stat_a[hartid] = s;
        printf("Tile %u | CASE A L2→L1 16KB | mean=%u min=%u max=%u CV=%u‰\n",
               hartid, s.mean, s.min, s.max, s.cv_permille);
    }

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // ================================================================
    // CASE B: Tile-to-tile ring shift (64 B, 1 hop, all tiles active)
    // Identical pattern to the ring-shift benchmark.
    // ================================================================

    uint32_t dest_x   = (x_id + 1) % MESH_X_TILES;
    uint32_t dest_id  = GET_ID(y_id, dest_x);
    uint32_t dest_base = dest_id * 0x00100000;

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    for (int t = 0; t < TRIAL_COUNT; t++) {
        uint32_t t0 = perf_get_cycles();

        // OBI→AXI (direction 1): local src → remote dst (l1_tmp of neighbor)
        idma_memcpy_1d(&idma_ctrl,
                       1,
                       (uint32_t)(dest_base + L1_TMP_OFFSET),
                       (uint32_t)l1_src,
                       BUF_SIZE_NOC);

        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);

        uint32_t t1 = perf_get_cycles();
        samples_b[t] = t1 - t0;

        // Brief sync between trials so data is settled before next send
        fsync_sync_global(&fsync_ctrl);
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    }

    {
        stat_t s;
        compute_stats(samples_b, TRIAL_COUNT, &s);
        tile_stat_b[hartid] = s;
        printf("Tile %u | CASE B tile→tile 64B | mean=%u min=%u max=%u CV=%u‰\n",
               hartid, s.mean, s.min, s.max, s.cv_permille);
    }

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // ================================================================
    // CASE C: Global fsync barrier
    // ================================================================

    for (int t = 0; t < TRIAL_COUNT; t++) {
        uint32_t t0 = perf_get_cycles();

        fsync_sync_global(&fsync_ctrl);
        eu_fsync_wait(&eu_ctrl, WAIT_MODE);

        uint32_t t1 = perf_get_cycles();
        samples_c[t] = t1 - t0;
    }

    {
        stat_t s;
        compute_stats(samples_c, TRIAL_COUNT, &s);
        tile_stat_c[hartid] = s;
        printf("Tile %u | CASE C global fsync | mean=%u min=%u max=%u CV=%u‰\n",
               hartid, s.mean, s.min, s.max, s.cv_permille);
    }

    // ----------------------------------------------------------------
    // Final global sync before tile 0 prints aggregate summary
    // ----------------------------------------------------------------

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // ----------------------------------------------------------------
    // Tile 0 aggregate summary
    // ----------------------------------------------------------------

    if (hartid == 0) {
        uint32_t num_tiles = MESH_X_TILES * MESH_Y_TILES;

        // Compute mesh-wide max CV for each case (worst-case jitter tile)
        uint32_t max_cv_a = 0, max_cv_b = 0, max_cv_c = 0;
        uint32_t sum_mean_a = 0, sum_mean_b = 0, sum_mean_c = 0;

        for (uint32_t i = 0; i < num_tiles; i++) {
            if (tile_stat_a[i].cv_permille > max_cv_a) max_cv_a = tile_stat_a[i].cv_permille;
            if (tile_stat_b[i].cv_permille > max_cv_b) max_cv_b = tile_stat_b[i].cv_permille;
            if (tile_stat_c[i].cv_permille > max_cv_c) max_cv_c = tile_stat_c[i].cv_permille;
            sum_mean_a += tile_stat_a[i].mean;
            sum_mean_b += tile_stat_b[i].mean;
            sum_mean_c += tile_stat_c[i].mean;
        }

        printf("\n=== REPEATABILITY SUMMARY (mesh %ux%u, %u trials) ===\n",
               MESH_X_TILES, MESH_Y_TILES, TRIAL_COUNT);
        printf("Case                    | mesh mean | worst CV(‰) | stable?\n");
        printf("A: L2→L1 16KB          |  %7u  |  %9u  | %s\n",
               sum_mean_a / num_tiles, max_cv_a,
               (max_cv_a < 10) ? "YES (CV<10‰)" : (max_cv_a < 50) ? "MARGINAL" : "NO - average required");
        printf("B: tile→tile 64B (NoC) |  %7u  |  %9u  | %s\n",
               sum_mean_b / num_tiles, max_cv_b,
               (max_cv_b < 10) ? "YES (CV<10‰)" : (max_cv_b < 50) ? "MARGINAL" : "NO - average required");
        printf("C: global fsync        |  %7u  |  %9u  | %s\n",
               sum_mean_c / num_tiles, max_cv_c,
               (max_cv_c < 10) ? "YES (CV<10‰)" : (max_cv_c < 50) ? "MARGINAL" : "NO - average required");

        printf("\nInterpretation:\n");
        printf("  CV < 10 per-mille  => single-run results are reliable\n");
        printf("  CV < 50 per-mille  => 4-8 repeats sufficient\n");
        printf("  CV >= 50 per-mille => 32+ repeats mandatory for valid comparison\n");
    }

    return 0;
}