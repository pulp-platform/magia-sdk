// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>
//
// L1 kernels called (on the CV32 control core) by the mgc-generated kernel src/mgc_conv_im2col.c.
// All pointer arguments are L1 addresses.
// Sizes are the kernel's arguments, read through the MGC_GLUE macros of mgc_conv_im2col.h.

#include <stdint.h>

#define MGC_GLUE
#include "mgc_conv_im2col.h"
#include "tile.h"
#include "im2col_fp16_spatz_params.h"
#include "eventunit.h"
#include "conv_im2col_spatz_task_bin.h"

/* Per-tile flag: set once Spatz has been set up on this tile (before its first task). */
static volatile uint8_t spatz_ready __attribute__((section(".tile_bss")));

/* Per-tile Spatz task parameters, preallocated in the per-tile L1. */
static struct {
    im2col_fp16_spatz_params_t im2col;
} spatz_params __attribute__((section(".tile_bss")));

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
    /* Compiler barrier: the parameter-block stores must be issued before Spatz is triggered. */
    asm volatile("" ::: "memory");
    spatz_run_task_with_params(task, (uint32_t)params);
}

/**
 * Starts the im2col of the CONV_CP output pixels of output row `row`, columns x0 .. x0 + CONV_CP - 1, into `col`
 * [CONV_CP x CONV_KKC] on this tile's Spatz; l1_spatz_wait1(col) waits for it. `x` holds the input window of the
 * W block loaded by the kernel: the CONV_K input rows lo .. lo + CONV_K - 1 that the row needs, columns
 * left .. left + CONV_WIN - 1, clamped to the image as below. The parameter block is rewritten at every start:
 * Spatz reads it only when the task starts, after the previous one has been waited.
 */
void l1_im2col_start_fp16_spatz(uint32_t col, uint32_t x, uint32_t row, uint32_t x0)
{
    im2col_fp16_spatz_params_t *p = &spatz_params.im2col;
    uint32_t lo, wx, left;

    /*
     * input window of the W block of x0 (CONV_WB output columns): those columns plus CONV_PAD on each side,
     * CONV_WIN in all, clamped inside the image like the rows
     */
    wx = x0 - x0 % CONV_WB;
    left = wx >= CONV_PAD ? wx - CONV_PAD : 0;
    if (left + CONV_WIN > CONV_W)
        left = CONV_W - CONV_WIN;

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
    /* the last block of a row of a W block may be partial: only its pixels inside the W block */
    p->cp     = wx + CONV_WB - x0 < CONV_CP ? wx + CONV_WB - x0 : CONV_CP;
    p->src_w  = CONV_WIN;
    p->src_x  = left;
    spatz_launch(IM2COL_FP16_SPATZ_TASK, p);
}

/**
 * Waits for the Spatz task started by an l1_*_start kernel, which writes x.
 */
void l1_spatz_wait1(uint32_t x)
{
    eu_wait_events(EU_SPATZ_DONE_MASK, WFE, 0);
}
