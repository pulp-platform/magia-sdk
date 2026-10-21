// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <francesco.conti@chips.it>
//
// L1 kernel called (synchronously, on the CV32 control core) by the mgc-generated test.c.
// All pointer arguments are L1 addresses.

#include <stdint.h>

#include "test.h"
#include "tile.h"
#include "eventunit.h"
#include "softmax_rows_fp16_spatz_params.h"
#include "unfused_attn_spatz_task_bin.h"

/* Per-tile flag: Spatz is set up on each tile before its first task. */
static volatile uint8_t spatz_ready __attribute__((section(".tile_bss")));

/* Per-tile Spatz task parameters, in the per-tile L1. */
static softmax_rows_fp16_spatz_params_t sm_params __attribute__((section(".tile_bss")));

/**
 * x[r, :] = softmax(x[r, :] / sqrt(PROJ_SIZE)) in place, for a `rows` x `cols` fp16 matrix, on
 * this tile's Spatz.
 */
void l1_softmax_rows_fp16_spatz(uint32_t x, uint32_t rows, uint32_t cols)
{
    softmax_rows_fp16_spatz_params_t *p = &sm_params;

    /* Lazy Spatz setup and fixed fields: the mgc-generated main does not know about Spatz. */
    if (!spatz_ready) {
        union { float16 f[2]; uint32_t u; } min_coef;
        float16 s = (float16)(1.0f / __builtin_sqrtf((float)PROJ_SIZE));

        min_coef.f[0] = (float16)(SOFTMAX_ROWS_EXP_MIN / (float)s);
        min_coef.f[1] = (float16)(SOFTMAX_ROWS_EXP_COEF * (float)s);
        p->min_coef   = min_coef.u;
        eu_spatz_init(0);
        spatz_init(SPATZ_BINARY_START);
        spatz_ready = 1;
    }
    p->shard_X = x;
    p->dims    = rows | (cols << 16);
    /* The block stores must land before Spatz is triggered. */
    asm volatile("" ::: "memory");
    spatz_run_task_with_params(SOFTMAX_ROWS_FP16_SPATZ_TASK, (uint32_t)p);
    eu_wait_events(EU_SPATZ_DONE_MASK, WFE, 0);
}
