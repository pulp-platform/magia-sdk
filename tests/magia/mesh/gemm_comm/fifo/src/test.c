// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "gemm_utils.h"
#include "fsync.h"
#include "idma.h"
#include "redmule.h"
#include "eventunit.h"
#include "utils/l1_fifo.h"

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

/* FIFO message type tags */
#define MATRIX_R1 0u
#define MATRIX_R2 1u
#define MATRIX_R3 2u

/*
 * 16 KB reserved at the start of each tile's L1 for the FIFO header,
 * linked-list nodes, and payload data. Workspace buffers follow after.
 */
#define FIFO_RESERVE_SIZE 0x4000u

/*
 * Rows pushed per fifo_push() call.
 * Set at build time via -DFIFO_BATCH_ROWS=N (default 1).
 */
#ifndef FIFO_BATCH_ROWS
#define FIFO_BATCH_ROWS 1
#endif

static const uint32_t gemm1_tiles[GEMM1_N_TILES] = {0, 1, 8, 9};

static const uint32_t gemm2_tiles[GEMM2_N_TILES] = {16, 17, 18, 19, 24, 25, 26, 27, 32, 33, 34, 35,
                                                    40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59};

static const uint32_t gemm3_tiles[GEMM3_N_TILES] = {2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15};

static const uint32_t gemm4_tiles[GEMM4_N_TILES] = {20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39,
                                                    44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63};

static int get_local_idx(uint32_t hartid, const uint32_t *tiles, uint32_t n_tiles)
{
    for (uint32_t i = 0; i < n_tiles; i++)
        if (tiles[i] == hartid)
            return (int)i;
    return -1;
}

static void get_row_range(uint32_t local_idx,
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

static void mem_set_zero(uint32_t o, uint32_t dim)
{
    for (uint32_t i = 0; i < dim; i++)
        mmio16(o + i * 2) = 0x0000;
}

/*
 * Copy nbytes from src to dst, both in L1 (local or remote).
 * Uses volatile word writes to ensure cross-tile visibility.
 */
static void copy_words(uint32_t dst, uint32_t src, uint32_t nbytes)
{
    uint32_t nwords = nbytes / 4;
    for (uint32_t i = 0; i < nwords; i++)
        ((volatile uint32_t *)dst)[i] = ((volatile uint32_t *)src)[i];
    for (uint32_t i = nwords * 4; i < nbytes; i++)
        ((volatile uint8_t *)dst)[i] = ((volatile uint8_t *)src)[i];
}

/**
 * GEMM chain test with FIFO-based out-of-order inter-tile communication.
 *
 * Same 4-GEMM chain as the L1 interlaced variant, but barriers between
 * Phase 1/2/3 are replaced by FIFO mailboxes. Producers push result rows
 * directly into consumers' FIFOs (allocated in the consumer's L1). Consumers
 * spin on their own FIFO and compute as soon as all required rows are present,
 * enabling out-of-order processing.
 *
 * Only two global barriers remain: a startup barrier (before any communication)
 * and a final barrier (before tile 0 validates o_out from L2).
 *
 * Phase 1 (parallel):
 *   GEMM1 (4 tiles):  R1[AxC] = M1[AxB] @ M2[BxC]
 *     -> push R1 rows to matching GEMM3 tiles' FIFOs (MATRIX_R1 tag)
 *   GEMM2 (24 tiles): R2[CxE] = M3[CxD] @ M4[DxE]
 *     -> push R2 rows to all GEMM3 tiles' FIFOs (MATRIX_R2 tag)
 *   GEMM4 (24 tiles): prefetch M5[ExF] from L2 into L1 workspace
 *     -> enter FIFO consumer loop immediately after (no Phase 1/2 barrier)
 *
 * Phase 2 (data-driven, no explicit barrier):
 *   GEMM3 (12 tiles): spin on own FIFO; for each received R1 batch, if R2 is
 *     already complete compute R3 immediately; when R2 becomes complete, compute
 *     all pending R1 batches. Push R3 rows to matching GEMM4 tiles' FIFOs (MATRIX_R3).
 *
 * Phase 3 (data-driven, no explicit barrier):
 *   GEMM4 (24 tiles): spin on own FIFO; for each received R3 batch, compute
 *     O = R3_batch @ M5 and DMA the result to L2.
 *
 * Final barrier + validation (same as reference).
 *
 * Requires 8x8 mesh (64 tiles).
 */
int main(void)
{
    /* 0. Initialization */
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

    /* Determine group membership for this tile (needed to size the FIFO) */
    int gemm1_idx = get_local_idx(hartid, gemm1_tiles, GEMM1_N_TILES);
    int gemm2_idx = get_local_idx(hartid, gemm2_tiles, GEMM2_N_TILES);
    int gemm3_idx = get_local_idx(hartid, gemm3_tiles, GEMM3_N_TILES);
    int gemm4_idx = get_local_idx(hartid, gemm4_tiles, GEMM4_N_TILES);

    /*
     * Initialize this tile's FIFO before the startup barrier.
     * Slot count and payload size depend on the tile's consumer role.
     */
    {
        uint32_t num_slots      = 0;
        uint32_t slot_data_size = 0;

        if (gemm3_idx >= 0) {
            uint32_t g3s, g3n;
            get_row_range((uint32_t)gemm3_idx, GEMM3_N_TILES, DIM_A, &g3s, &g3n);
            uint32_t r1_payload = FIFO_BATCH_ROWS * DIM_C * 2;
            uint32_t r2_payload = FIFO_BATCH_ROWS * DIM_E * 2;
            num_slots           = g3n + DIM_C;
            slot_data_size      = r1_payload > r2_payload ? r1_payload : r2_payload;
        } else if (gemm4_idx >= 0) {
            uint32_t g4s, g4n;
            get_row_range((uint32_t)gemm4_idx, GEMM4_N_TILES, DIM_A, &g4s, &g4n);
            num_slots      = g4n;
            slot_data_size = FIFO_BATCH_ROWS * DIM_E * 2;
        }

        fifo_init(hartid, num_slots, slot_data_size);
    }

    /*
     * Startup barrier: ensure all tiles finish crt0.S BSS zeroing (which
     * touches global output buffers) and FIFO initialization before any
     * tile begins pushing into another tile's FIFO.
     */
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /*
     * L1 workspace starts at FIFO_RESERVE_SIZE bytes past the tile base.
     * The first FIFO_RESERVE_SIZE bytes are reserved for the FIFO header
     * and slot array written by remote pushers.
     */
    uint32_t ws = l1_tile_base + FIFO_RESERVE_SIZE;

    /* ------------------------------------------------------------------ */
    /* GEMM1: R1 = M1 @ M2, push R1 rows to GEMM3 FIFOs        */
    /* ------------------------------------------------------------------ */
    if (gemm1_idx >= 0) {
        uint32_t start_row, num_rows;
        get_row_range((uint32_t)gemm1_idx, GEMM1_N_TILES, DIM_A, &start_row, &num_rows);

        if (num_rows > 0) {
            /* Workspace layout: [M1_slice | M2 | R1_slice] */
            uint32_t obi_m1 = ws;
            uint32_t obi_m2 = obi_m1 + num_rows * DIM_B * 2;
            uint32_t obi_r1 = obi_m2 + DIM_B * DIM_C * 2;

            /* Load M1 slice from L2 */
            idma_memcpy_1d(&idma_ctrl,
                           0,
                           (uint32_t)m1_inp + start_row * DIM_B * 2,
                           obi_m1,
                           num_rows * DIM_B * 2);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

            /* Load full M2 from L2 */
            idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m2_inp, obi_m2, DIM_B * DIM_C * 2);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

            /* Compute R1_slice = M1_slice @ M2 */
            mem_set_zero(obi_r1, num_rows * DIM_C);
            redmule_gemm(&redmule_ctrl,
                         obi_m1,
                         obi_m2,
                         obi_r1,
                         (uint16_t)num_rows,
                         (uint16_t)DIM_B,
                         (uint16_t)DIM_C);
            eu_redmule_wait(&eu_ctrl, WAIT_MODE);

            /*
             * Push R1 rows to each GEMM3 tile that owns overlapping rows.
             * Pushes are batched in FIFO_BATCH_ROWS increments.
             * row_index tag = global row of the first row in the batch.
             */
            for (uint32_t j = 0; j < GEMM3_N_TILES; j++) {
                printf("Tile %u step %u: Checking overlap with GEMM3 tile %u\n",
                       hartid,
                       j,
                       gemm3_tiles[j]);

                uint32_t g3_start, g3_nrows;
                get_row_range(j, GEMM3_N_TILES, DIM_A, &g3_start, &g3_nrows);
                if (g3_nrows == 0)
                    continue;

                printf("Tile %u step %u: GEMM3 tile %u has rows [%u, %u)\n",
                       hartid,
                       j,
                       gemm3_tiles[j],
                       g3_start,
                       g3_start + g3_nrows);

                /* Overlap between this tile's R1 rows and GEMM3 tile j's rows */
                uint32_t ov_start = start_row > g3_start ? start_row : g3_start;
                uint32_t ov_end   = (start_row + num_rows) < (g3_start + g3_nrows)
                                        ? (start_row + num_rows)
                                        : (g3_start + g3_nrows);

                printf("Tile %u step %u: Overlap with GEMM3 tile %u is rows [%u, %u)\n",
                       hartid,
                       j,
                       gemm3_tiles[j],
                       ov_start,
                       ov_end);

                if (ov_start >= ov_end)
                    continue;

                for (uint32_t r = ov_start; r < ov_end; r += FIFO_BATCH_ROWS) {
                    printf("Tile %u step %u: Pushing R1 rows [%u, %u) to GEMM3 tile %u\n",
                           hartid,
                           j,
                           r,
                           (r + FIFO_BATCH_ROWS <= ov_end) ? (r + FIFO_BATCH_ROWS) : ov_end,
                           gemm3_tiles[j]);

                    uint32_t batch =
                        (r + FIFO_BATCH_ROWS <= ov_end) ? FIFO_BATCH_ROWS : (ov_end - r);
                    uint32_t src_off = (r - start_row) * DIM_C * 2;

                    printf("Tile %u step %u: src_off=%u bytes, batch size=%u rows (%u bytes)\n",
                           hartid,
                           j,
                           src_off,
                           batch,
                           batch * DIM_C * 2);

                    uint32_t slot_idx = r - g3_start;
                    fifo_push(gemm3_tiles[j],
                              slot_idx,
                              (void *)(obi_r1 + src_off),
                              batch * DIM_C * 2,
                              MATRIX_R1,
                              r);

                    printf("Tile %u step %u: Pushed R1 rows [%u, %u) to GEMM3 tile %u\n",
                           hartid,
                           j,
                           r,
                           (r + batch <= ov_end) ? (r + batch) : ov_end,
                           gemm3_tiles[j]);
                }
            }

            printf("Tile %u: GEMM1 pushes to GEMM3 FIFOs done\n", hartid);
        }
    }

    /* ------------------------------------------------------------------ */
    /* GEMM2: R2 = M3 @ M4, push R2 rows to all GEMM3 FIFOs    */
    /* ------------------------------------------------------------------ */
    if (gemm2_idx >= 0) {
        uint32_t start_row, num_rows;
        get_row_range((uint32_t)gemm2_idx, GEMM2_N_TILES, DIM_C, &start_row, &num_rows);

        if (num_rows > 0) {
            /* Workspace layout: [M3_slice | M4 | R2_slice] */
            uint32_t obi_m3 = ws;
            uint32_t obi_m4 = obi_m3 + num_rows * DIM_D * 2;
            uint32_t obi_r2 = obi_m4 + DIM_D * DIM_E * 2;

            /* Load M3 slice from L2 */
            idma_memcpy_1d(&idma_ctrl,
                           0,
                           (uint32_t)m3_inp + start_row * DIM_D * 2,
                           obi_m3,
                           num_rows * DIM_D * 2);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

            /* Load full M4 from L2 */
            idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m4_inp, obi_m4, DIM_D * DIM_E * 2);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

            /* Compute R2_slice = M3_slice @ M4 */
            mem_set_zero(obi_r2, num_rows * DIM_E);
            redmule_gemm(&redmule_ctrl,
                         obi_m3,
                         obi_m4,
                         obi_r2,
                         (uint16_t)num_rows,
                         (uint16_t)DIM_D,
                         (uint16_t)DIM_E);
            eu_redmule_wait(&eu_ctrl, WAIT_MODE);

            /*
             * Push R2 rows to every GEMM3 tile (all need the full R2).
             * row_index = global row in the full R2 matrix (0..DIM_C-1).
             */
            for (uint32_t r = 0; r < num_rows; r += FIFO_BATCH_ROWS) {
                uint32_t batch =
                    (r + FIFO_BATCH_ROWS <= num_rows) ? FIFO_BATCH_ROWS : (num_rows - r);
                uint32_t src_off = r * DIM_E * 2;
                for (uint32_t j = 0; j < GEMM3_N_TILES; j++) {
                    /* R2 slots start after R1 slots in the consumer's FIFO */
                    uint32_t g3s_j, g3n_j;
                    get_row_range(j, GEMM3_N_TILES, DIM_A, &g3s_j, &g3n_j);
                    uint32_t slot_idx = g3n_j + (start_row + r);
                    fifo_push(gemm3_tiles[j],
                              slot_idx,
                              (void *)(obi_r2 + src_off),
                              batch * DIM_E * 2,
                              MATRIX_R2,
                              start_row + r);
                }
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* GEMM3: out-of-order FIFO consumer + R3 producer          */
    /* ------------------------------------------------------------------ */
    if (gemm3_idx >= 0) {
        uint32_t start_row, num_rows;
        get_row_range((uint32_t)gemm3_idx, GEMM3_N_TILES, DIM_A, &start_row, &num_rows);

        /*
         * Workspace layout: [R2 (full DIM_C x DIM_E) | R3_slice (num_rows x DIM_E)]
         * R1 data is used directly from FIFO payload pointers (no separate R1 workspace).
         */
        uint32_t obi_r2 = ws;
        uint32_t obi_r3 = obi_r2 + DIM_C * DIM_E * 2;

        /*
         * Tracking state for out-of-order processing.
         * Arrays indexed by local row (0..DIM_A-1); only [0..num_rows-1] are used.
         * r1_received[lr]  : 1 once the batch starting at local row lr has arrived
         * r1_data_ptrs[lr] : FIFO payload pointer for the batch at local row lr
         * r3_computed[lr]  : 1 once R3 for that batch has been computed and pushed
         */
        uint8_t r1_received[DIM_A];
        uint32_t r1_data_ptrs[DIM_A];
        uint8_t r3_computed[DIM_A];

        for (uint32_t i = 0; i < DIM_A; i++) {
            r1_received[i]  = 0;
            r1_data_ptrs[i] = 0;
            r3_computed[i]  = 0;
        }

        uint32_t r2_rows_received = 0;
        uint32_t total_r3_done    = 0;

        /*
         * Spin until all R3 rows for this tile have been computed and pushed.
         * GEMM3 tiles with num_rows == 0 skip this loop entirely.
         */
        while (total_r3_done < num_rows) {
            uint32_t data_ptr, data_size, matrix_id, row_index;

            /* Spin until a FIFO message arrives */
            while (!fifo_pop(hartid, &data_ptr, &data_size, &matrix_id, &row_index))
                ;

            if (matrix_id == MATRIX_R1) {
                /*
                 * Received a batch of R1 rows. Record the payload pointer and mark received.
                 * The batch starts at global row row_index; local_row = row_index - start_row.
                 */
                uint32_t local_row      = row_index - start_row;
                r1_data_ptrs[local_row] = data_ptr;
                r1_received[local_row]  = 1;

                if (r2_rows_received == DIM_C) {
                    /* R2 already complete: compute R3 for this batch immediately */
                    uint32_t batch_rows   = data_size / (DIM_C * 2);
                    uint32_t obi_r3_batch = obi_r3 + local_row * DIM_E * 2;

                    mem_set_zero(obi_r3_batch, batch_rows * DIM_E);
                    redmule_gemm(&redmule_ctrl,
                                 data_ptr,
                                 obi_r2,
                                 obi_r3_batch,
                                 (uint16_t)batch_rows,
                                 (uint16_t)DIM_C,
                                 (uint16_t)DIM_E);
                    eu_redmule_wait(&eu_ctrl, WAIT_MODE);

                    /* Push R3 batch to overlapping GEMM4 tiles' FIFOs */
                    uint32_t global_row = row_index;
                    for (uint32_t k = 0; k < GEMM4_N_TILES; k++) {
                        uint32_t g4_start, g4_nrows;
                        get_row_range(k, GEMM4_N_TILES, DIM_A, &g4_start, &g4_nrows);
                        if (g4_nrows == 0)
                            continue;

                        uint32_t ov_start = global_row > g4_start ? global_row : g4_start;
                        uint32_t ov_end   = (global_row + batch_rows) < (g4_start + g4_nrows)
                                                ? (global_row + batch_rows)
                                                : (g4_start + g4_nrows);
                        if (ov_start >= ov_end)
                            continue;

                        uint32_t ov_count = ov_end - ov_start;
                        uint32_t src_off  = (ov_start - global_row) * DIM_E * 2;
                        uint32_t slot_idx = ov_start - g4_start;
                        fifo_push(gemm4_tiles[k],
                                  slot_idx,
                                  (void *)(obi_r3_batch + src_off),
                                  ov_count * DIM_E * 2,
                                  MATRIX_R3,
                                  ov_start);
                    }

                    r3_computed[local_row] = 1;
                    total_r3_done += batch_rows;
                }
                /* else: R2 not yet complete; this batch will be processed when R2 arrives */

            } else if (matrix_id == MATRIX_R2) {
                /*
                 * Received a batch of R2 rows. Copy into the contiguous R2 workspace
                 * at the correct row offset. row_index is the global R2 row (0..DIM_C-1).
                 */
                uint32_t batch_rows = data_size / (DIM_E * 2);
                copy_words(obi_r2 + row_index * DIM_E * 2, data_ptr, data_size);
                r2_rows_received += batch_rows;

                if (r2_rows_received == DIM_C) {
                    /*
                     * R2 just became complete. Process all R1 batches that arrived earlier
                     * and haven't been computed yet.
                     */
                    for (uint32_t lr = 0; lr < num_rows; lr += FIFO_BATCH_ROWS) {
                        if (!r1_received[lr] || r3_computed[lr])
                            continue;

                        uint32_t remaining = num_rows - lr;
                        uint32_t batch_r3 =
                            (remaining < FIFO_BATCH_ROWS) ? remaining : FIFO_BATCH_ROWS;
                        uint32_t r1_payload   = r1_data_ptrs[lr];
                        uint32_t obi_r3_batch = obi_r3 + lr * DIM_E * 2;

                        mem_set_zero(obi_r3_batch, batch_r3 * DIM_E);
                        redmule_gemm(&redmule_ctrl,
                                     r1_payload,
                                     obi_r2,
                                     obi_r3_batch,
                                     (uint16_t)batch_r3,
                                     (uint16_t)DIM_C,
                                     (uint16_t)DIM_E);
                        eu_redmule_wait(&eu_ctrl, WAIT_MODE);

                        uint32_t global_row = start_row + lr;
                        for (uint32_t k = 0; k < GEMM4_N_TILES; k++) {
                            uint32_t g4_start, g4_nrows;
                            get_row_range(k, GEMM4_N_TILES, DIM_A, &g4_start, &g4_nrows);
                            if (g4_nrows == 0)
                                continue;

                            uint32_t ov_start = global_row > g4_start ? global_row : g4_start;
                            uint32_t ov_end   = (global_row + batch_r3) < (g4_start + g4_nrows)
                                                    ? (global_row + batch_r3)
                                                    : (g4_start + g4_nrows);
                            if (ov_start >= ov_end)
                                continue;

                            uint32_t ov_count = ov_end - ov_start;
                            uint32_t src_off  = (ov_start - global_row) * DIM_E * 2;
                            uint32_t slot_idx = ov_start - g4_start;
                            fifo_push(gemm4_tiles[k],
                                      slot_idx,
                                      (void *)(obi_r3_batch + src_off),
                                      ov_count * DIM_E * 2,
                                      MATRIX_R3,
                                      ov_start);
                        }

                        r3_computed[lr] = 1;
                        total_r3_done += batch_r3;
                    }
                }
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* GEMM4: prefetch M5 into L1, then enter FIFO consumer     */
    /* ------------------------------------------------------------------ */
    if (gemm4_idx >= 0) {
        uint32_t start_row, num_rows;
        get_row_range((uint32_t)gemm4_idx, GEMM4_N_TILES, DIM_A, &start_row, &num_rows);

        if (num_rows > 0) {
            /*
             * Workspace layout: [M5 | O_slice]
             * M5 is placed first so Phase 3 only needs to know ws to find it.
             */
            uint32_t obi_m5 = ws;
            uint32_t obi_o  = obi_m5 + DIM_E * DIM_F * 2;

            /* Prefetch full M5 from L2 */
            idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m5_inp, obi_m5, DIM_E * DIM_F * 2);
            eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

            /*
             * FIFO consumer loop (Phase 3, data-driven, no explicit barrier).
             * Spin until all expected R3 rows (== num_rows) have been processed.
             */
            uint32_t total_o_done = 0;
            while (total_o_done < num_rows) {
                uint32_t data_ptr, data_size, matrix_id, row_index;

                /* Spin until a message arrives */
                while (!fifo_pop(hartid, &data_ptr, &data_size, &matrix_id, &row_index))
                    ;

                /* Only MATRIX_R3 messages are expected by GEMM4 tiles */
                if (matrix_id != MATRIX_R3)
                    continue;

                uint32_t batch_rows  = data_size / (DIM_E * 2);
                uint32_t local_row   = row_index - start_row;
                uint32_t obi_o_batch = obi_o + local_row * DIM_F * 2;

                /* Compute O_batch = R3_batch @ M5; use payload directly as R3 input */
                mem_set_zero(obi_o_batch, batch_rows * DIM_F);
                redmule_gemm(&redmule_ctrl,
                             data_ptr,
                             obi_m5,
                             obi_o_batch,
                             (uint16_t)batch_rows,
                             (uint16_t)DIM_E,
                             (uint16_t)DIM_F);
                eu_redmule_wait(&eu_ctrl, WAIT_MODE);

                /* Write O batch to L2 */
                idma_memcpy_1d(&idma_ctrl,
                               1,
                               (uint32_t)o_out + row_index * DIM_F * 2,
                               obi_o_batch,
                               batch_rows * DIM_F * 2);
                eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);

                total_o_done += batch_rows;
            }
        }
    }

    /* ------------------------------------------------------------------ */
    /* Final barrier: wait for all tiles to finish before validation       */
    /* ------------------------------------------------------------------ */
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* Validation: tile 0 checks O against golden */
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
