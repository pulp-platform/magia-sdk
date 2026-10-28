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
 * [CONV_CP x CONV_KKC] on this tile's Spatz; `x` holds the CONV_K input rows lo .. lo + CONV_K - 1 that the row needs,
 * the window that the test loads (clamped to the image, as below). l1_im2col_wait(col) waits for it. With the share
 * option, this tile builds only its part of the block, pixels x_id * SHARE_CP onwards. The parameter block is
 * rewritten at every start: Spatz reads it only when the task starts, after the previous one is waited.
 */
void l1_im2col_start_fp16_spatz(uint32_t col, uint32_t x, uint32_t row, uint32_t x0)
{
    im2col_fp16_spatz_params_t *p = &spatz_params.im2col;
    uint32_t lo;

    if (row + CONV_PAD >= CONV_H)
        lo = CONV_H - CONV_K;
    else if (row >= CONV_PAD)
        lo = row - CONV_PAD;
    else
        lo = 0;

    p->src    = x;
    p->dst    = col;
    p->height = CONV_H;
    p->width  = CONV_W;
    p->cin    = CONV_CIN;
    p->ksize  = CONV_K;
    p->pad    = CONV_PAD;
    p->row    = row;
    p->lo     = lo;
    p->x0     = x0;
    p->cp     = CONV_CP;
#if OPT_SHARE
    {
        uint32_t part = GET_X_ID(get_hartid()) * SHARE_CP;

        p->dst     = col + part * CONV_KKC * 2;
        p->x0      = x0 + part;
        p->cp      = SHARE_CHUNK;
        share_part = p->dst;
        share_x0   = p->x0;
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

    for (uint32_t c = 0; c < SHARE_NB_CHUNKS; c++) {
        uint32_t first = c * SHARE_CHUNK;
        uint32_t n     = SHARE_CP - first < SHARE_CHUNK ? SHARE_CP - first : SHARE_CHUNK;
        uint32_t addr  = share_part + first * CONV_KKC * 2;

        eu_wait_events(EU_SPATZ_DONE_MASK, WFE, 0);
        if (c + 1 < SHARE_NB_CHUNKS) {
            uint32_t next = first + SHARE_CHUNK;

            p->dst = share_part + next * CONV_KKC * 2;
            p->x0  = share_x0 + next;
            p->cp  = SHARE_CP - next < SHARE_CHUNK ? SHARE_CP - next : SHARE_CHUNK;
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

