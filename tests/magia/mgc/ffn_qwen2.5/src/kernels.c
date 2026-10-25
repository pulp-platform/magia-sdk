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
#include "swiglu_fast_fp16_spatz_params.h"
#include "eventunit.h"
#include "ffn_qwen2_5_spatz_task_bin.h"

/* Per-tile flag: Spatz is set up on each tile before its first task. */
static volatile uint8_t spatz_ready __attribute__((section(".tile_bss")));

/* Per-tile Spatz task parameters, preallocated in the per-tile L1. */
static struct {
    swiglu_fast_fp16_spatz_params_t swiglu;
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
 * Starts g[i] = silu(g[i]) * u[i] for `n` fp16 elements, in place, on this tile's Spatz;
 * l1_spatz_wait1(g) waits for it. The parameter block is rewritten at every start: the double-buffered
 * chunks alternate, and Spatz reads it only when the task starts (after the previous one is waited).
 */
void l1_swiglu_start_fp16_spatz(uint32_t g, uint32_t u, uint32_t n)
{
    swiglu_fast_fp16_spatz_params_t *p = &spatz_params.swiglu;

    p->shard_G = g;
    p->shard_U = u;
    p->shard_Y = g;
    p->len     = n;
    spatz_launch(SWIGLU_FAST_FP16_SPATZ_TASK, p);
}

/**
 * Waits for the Spatz task started by an l1_*_start kernel, which writes x.
 */
void l1_spatz_wait1(uint32_t x)
{
    eu_wait_events(EU_SPATZ_DONE_MASK, WFE, 0);
}
