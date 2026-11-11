// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "softmax_rows_fp16_spatz.h"
#include "softmax_rows_fp16_spatz_params.h"
#include "softmax_rows_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "softmax_rows_fp16_spatz"

/* fp16 bits of the exp clamp MIN / s | COEF * s << 16, for the scale s. */
static uint32_t min_coef(float16 s)
{
    union {
        float16 f[2];
        uint32_t u;
    } mc;

    mc.f[0] = (float16)(SOFTMAX_ROWS_EXP_MIN / (float)s);
    mc.f[1] = (float16)(SOFTMAX_ROWS_EXP_COEF * (float)s);

    return mc.u;
}

/* This tile's rows [*row_start, *row_start + *row_len) out of rows. */
static void tile_rows(uint32_t rows, uint32_t *row_start, uint32_t *row_len)
{
    uint32_t elems = rows / NUM_HARTS;
    uint32_t left  = rows % NUM_HARTS;

    *row_start = HID * elems + (HID < left ? HID : left);
    *row_len   = elems + (HID < left ? 1 : 0);
}

static int alloc_l1(void **params, float16 s, uint32_t rows, uint32_t cols)
{
    volatile softmax_rows_fp16_spatz_params_t *sm_params;
    uintptr_t shard_X;
    uint32_t row_start;
    uint32_t row_len;

    tile_rows(rows, &row_start, &row_len);

    l1_alloc_init();

    sm_params = l1_alloc(sizeof(softmax_rows_fp16_spatz_params_t));
    if (!sm_params)
        return ENOMEM;

    shard_X = (uintptr_t)l1_alloc(row_len * cols * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    sm_params->shard_X  = shard_X;
    sm_params->dims     = row_len | (cols << 16);
    sm_params->min_coef = min_coef(s);

    *params = (void *)sm_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, uint32_t rows)
{
    volatile softmax_rows_fp16_spatz_params_t *sm_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    sm_params = (volatile softmax_rows_fp16_spatz_params_t *)params;
    cols      = sm_params->dims >> 16;
    tile_rows(rows, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    /* This tile's rows are contiguous in X. */
    idma_memcpy_1d(0,
                   (uint32_t)(X + row_start * cols),
                   (uint32_t)sm_params->shard_X,
                   row_len * cols * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    spatz_run_task_with_params(SOFTMAX_ROWS_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

void MAGIA_softmax_rows_fp16_spatz_l1(float16 *X, float16 s, uint32_t rows, uint32_t cols)
{
    volatile softmax_rows_fp16_spatz_params_t params;
    int ret;

    /* Field by field: GCC hits an internal error on this 3-word volatile initializer. */
    params.shard_X  = (uintptr_t)X;
    params.dims     = rows | (cols << 16);
    params.min_coef = min_coef(s);

    ret = offload_spatz_task((void *)&params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
    }
}

static int store_result(void *params, float16 *X, uint32_t rows)
{
    volatile softmax_rows_fp16_spatz_params_t *sm_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    sm_params = (volatile softmax_rows_fp16_spatz_params_t *)params;
    cols      = sm_params->dims >> 16;
    tile_rows(rows, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    idma_memcpy_1d(1,
                   (uint32_t)(X + row_start * cols),
                   (uint32_t)sm_params->shard_X,
                   row_len * cols * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    return 0;
}

void MAGIA_softmax_rows_fp16_spatz(float16 *X, float16 s, uint32_t rows, uint32_t cols)
{
    int ret;
    volatile softmax_rows_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, s, rows, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X, rows);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = offload_spatz_task((void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = store_result((void *)params, X, rows);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
