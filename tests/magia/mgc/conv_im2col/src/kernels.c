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
#include "conv_im2col_spatz_task_bin.h"

/* Per-tile flag: Spatz is set up on each tile before its first task. */
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
    /* The block stores must land before Spatz is triggered. */
    asm volatile("" ::: "memory");
    spatz_run_task_with_params(task, (uint32_t)params);
}

/**
 * Starts the im2col of the CONV_CP output pixels of output row `row`, columns x0 .. x0 + CONV_CP - 1, into `col`
 * [CONV_CP x CONV_KKC] on this tile's Spatz; `x` holds the CONV_K input rows lo .. lo + CONV_K - 1 that the row needs,
 * the window that the test loads (clamped to the image, as below). l1_spatz_wait1(col) waits for it. The parameter
 * block is rewritten at every start: Spatz reads it only when the task starts, after the previous one is waited.
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
    spatz_launch(IM2COL_FP16_SPATZ_TASK, p);
}

/**
 * Waits for the Spatz task started by an l1_*_start kernel, which writes x.
 */
void l1_spatz_wait1(uint32_t x)
{
    eu_wait_events(EU_SPATZ_DONE_MASK, WFE, 0);
}
