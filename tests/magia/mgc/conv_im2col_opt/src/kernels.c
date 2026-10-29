// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>
//
// L1 kernels called (on the CV32 control core) by the mgc-generated test.c.
// All pointer arguments are L1 addresses.

#include <stdint.h>

#include "test.h"
#include "tile.h"
#include "im2col_fp16_spatz_params.h"
#include "eventunit.h"
#include "mg_idma.h"
#include "conv_im2col_opt_spatz_task_bin.h"

/* Per-tile flag: Spatz is set up on each tile before its first task. */
static volatile uint8_t spatz_ready __attribute__((section(".tile_bss")));

/* Per-tile Spatz task parameters, preallocated in the per-tile L1. */
static struct {
    im2col_fp16_spatz_params_t im2col;
} spatz_params __attribute__((section(".tile_bss")));

/* The tiles of a mesh row share the im2col: each builds CONV_CP / MESH_X_TILES pixels of the block and multicasts
   them to the others (on a mesh with one tile per row, the block is built whole). */
#define OPT_SHARE (MESH_X_TILES > 1)
#define SHARE_CP  (CONV_CP / MESH_X_TILES)
/* The part is built in Spatz tasks of CONV_SHARE_CHUNK pixels (0: one task), and each chunk is multicast while
   Spatz builds the next one. */
#if CONV_SHARE_CHUNK > 0 && CONV_SHARE_CHUNK < SHARE_CP
#define SHARE_CHUNK CONV_SHARE_CHUNK
#else
#define SHARE_CHUNK SHARE_CP
#endif
#define SHARE_NB_CHUNKS ((SHARE_CP + SHARE_CHUNK - 1) / SHARE_CHUNK)

/* Share: the L1 address and first output column of this tile's part of the block being built, used by
   l1_im2col_wait to start the next chunks. */
static uint32_t share_part __attribute__((section(".tile_bss")));
static uint32_t share_x0 __attribute__((section(".tile_bss")));
/* Share: pixels of this tile's part inside the W block (the last block of a row of a W block may be partial, and
   a part past its end has none). */
static uint32_t share_n __attribute__((section(".tile_bss")));

/**
 * Lazy Spatz setup on this tile (event unit line, binary and clock): the mgc-generated main
 * does not know about Spatz.
 */
static void spatz_setup(void)
{
    if (spatz_ready)
        return;
    eu_spatz_init(0);
    spatz_init(SPATZ_BINARY_START);
    spatz_ready = 1;
}

/* Starts `task` on this tile's Spatz with the parameter block `params`, without waiting. */
static inline void spatz_launch(uint32_t task, void *params)
{
    spatz_setup();
    /* The block stores must land before Spatz is triggered. */
    asm volatile("" ::: "memory");
    spatz_run_task_with_params(task, (uint32_t)params);
}

/**
 * Starts the im2col of the CONV_CP output pixels of output row `row`, columns x0 .. x0 + CONV_CP - 1, into `col`
 * [CONV_CP x CONV_KKC] on this tile's Spatz; `x` holds the CONV_K input rows lo .. lo + CONV_K - 1 that the row needs
 * (from row * CONV_S - CONV_PAD), columns left .. left + CONV_WIN - 1 of them, the window of the W block that the
 * test loads (clamped to the image, as below). l1_im2col_wait(col) waits for it. With the share
 * option, this tile builds only its part of the block, pixels x_id * SHARE_CP onwards. The parameter block is
 * rewritten at every start: Spatz reads it only when the task starts, after the previous one is waited.
 */
void l1_im2col_start_fp16_spatz(uint32_t col, uint32_t x, uint32_t row, uint32_t x0)
{
    im2col_fp16_spatz_params_t *p = &spatz_params.im2col;
    uint32_t lo, wx, left;

    /* the input window of the W block of x0 (CONV_WB output columns): the input columns of their patches, from
       wx * CONV_S - CONV_PAD, CONV_WIN in all, clamped inside the image as the rows */
    wx = x0 - x0 % CONV_WB;
    left = wx * CONV_S >= CONV_PAD ? wx * CONV_S - CONV_PAD : 0;
    if (left + CONV_WIN > CONV_W)
        left = CONV_W - CONV_WIN;

    if (row * CONV_S + CONV_PAD >= CONV_H)
        lo = CONV_H - CONV_K;
    else if (row * CONV_S >= CONV_PAD)
        lo = row * CONV_S - CONV_PAD;
    else
        lo = 0;

    p->src    = x;
    p->dst    = col;
    p->height = CONV_H;
    p->width  = CONV_W;
    p->cin    = CONV_CIN;
    p->ksize  = CONV_K;
    p->stride = CONV_S;
    p->pad    = CONV_PAD;
    p->row    = row;
    p->lo     = lo;
    p->x0     = x0;
    /* the last block of a row of a W block may be partial: only its pixels inside the W block */
    p->cp     = wx + CONV_WB - x0 < CONV_CP ? wx + CONV_WB - x0 : CONV_CP;
    p->src_w  = CONV_WIN;
    p->src_x  = left;
#if OPT_SHARE
    {
        uint32_t part = GET_X_ID(get_hartid()) * SHARE_CP;

        share_n    = p->cp <= part ? 0 : p->cp - part < SHARE_CP ? p->cp - part : SHARE_CP;
        p->dst     = col + part * CONV_KKC * 2;
        p->x0      = x0 + part;
        p->cp      = share_n < SHARE_CHUNK ? share_n : SHARE_CHUNK;
        share_part = p->dst;
        share_x0   = p->x0;
        if (share_n == 0)
            return;
    }
#endif
    spatz_launch(IM2COL_FP16_SPATZ_TASK, p);
}

/**
 * Waits for the im2col started by l1_im2col_start_fp16_spatz, which writes col. With the share option, the part of
 * this tile is built chunk by chunk: once Spatz has a chunk, it starts the next one and the chunk is sent to the same
 * place in `col` on the other tiles of the row; then all the sends are waited. The barrier that ends the pipeline
 * step (sync=ROW) makes the parts of the other tiles complete before RedMulE uses the block. `col` is at the same L1
 * offset on every tile (it is the first buffer of the test).
 */
void l1_im2col_wait(uint32_t col)
{
#if OPT_SHARE
    im2col_fp16_spatz_params_t *p = &spatz_params.im2col;
    mg_event_t ev;

    uint32_t nb_chunks = (share_n + SHARE_CHUNK - 1) / SHARE_CHUNK;

    if (nb_chunks == 0)
        return; /* nothing of this tile's part inside the W block */
    for (uint32_t c = 0; c < nb_chunks; c++) {
        uint32_t first = c * SHARE_CHUNK;
        uint32_t n     = share_n - first < SHARE_CHUNK ? share_n - first : SHARE_CHUNK;
        uint32_t addr  = share_part + first * CONV_KKC * 2;

        eu_wait_events(EU_SPATZ_DONE_MASK, WFE, 0);
        if (c + 1 < nb_chunks) {
            uint32_t next = first + SHARE_CHUNK;

            p->dst = share_part + next * CONV_KKC * 2;
            p->x0  = share_x0 + next;
            p->cp  = share_n - next < SHARE_CHUNK ? share_n - next : SHARE_CHUNK;
            spatz_launch(IM2COL_FP16_SPATZ_TASK, p);
        }
        /* same-direction iDMA jobs complete in order: waiting for the last one waits for all of them */
        mg_idma_mcast_row(WFE, 1, addr, addr, n * CONV_KKC * 2, &ev, NULL);
    }
    mg_idma_wait(1, WFE, &ev);
#else
    eu_wait_events(EU_SPATZ_DONE_MASK, WFE, 0);
#endif
}

