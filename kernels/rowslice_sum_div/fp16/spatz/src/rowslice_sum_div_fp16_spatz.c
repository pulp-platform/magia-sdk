// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "rowslice_sum_div_fp16_spatz.h"
#include "rowslice_sum_div_fp16_spatz_params.h"
#include "rowslice_sum_div_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "rowslice_sum_div_fp16_spatz"

/* This tile's rows [*row_start, *row_start + *row_len) out of rows. */
static void tile_rows(uint32_t rows, uint32_t *row_start, uint32_t *row_len)
{
    uint32_t elems = rows / NUM_HARTS;
    uint32_t left  = rows % NUM_HARTS;

    *row_start = HID * elems + (HID < left ? HID : left);
    *row_len   = elems + (HID < left ? 1 : 0);
}

static int alloc_l1(void **params, uint32_t n, uint32_t own, uint32_t rows, uint32_t cols)
{
    volatile rowslice_sum_div_fp16_spatz_params_t *rd_params;
    uintptr_t shard_Y;
    uintptr_t shard_O;
    uintptr_t shard_G;
    uintptr_t shard_S;
    uint32_t row_start;
    uint32_t row_len;

    tile_rows(rows, &row_start, &row_len);

    l1_alloc_init();

    rd_params = l1_alloc(sizeof(rowslice_sum_div_fp16_spatz_params_t));
    if (!rd_params)
        return ENOMEM;

    shard_Y = (uintptr_t)l1_alloc(row_len * cols * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    shard_O = (uintptr_t)l1_alloc(row_len * cols * sizeof(float16));
    if (!shard_O)
        return ENOMEM;

    shard_G = (uintptr_t)l1_alloc(n * row_len * cols * sizeof(float16));
    if (!shard_G)
        return ENOMEM;

    shard_S = (uintptr_t)l1_alloc(row_len * sizeof(float16));
    if (!shard_S)
        return ENOMEM;

    rd_params->shard_Y = shard_Y;
    rd_params->shard_O = shard_O;
    rd_params->shard_G = shard_G;
    rd_params->shard_S = shard_S;
    rd_params->dims    = row_len | (cols << 16);
    rd_params->slices  = n | (own << 16);

    *params = (void *)rd_params;

    return 0;
}

static int
init_input_params(void *params, const float16 *O, const float16 *G, const float16 *S, uint32_t rows)
{
    volatile rowslice_sum_div_fp16_spatz_params_t *rd_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;
    uint32_t n;

    rd_params = (volatile rowslice_sum_div_fp16_spatz_params_t *)params;
    cols      = rd_params->dims >> 16;
    n         = rd_params->slices & 0xffff;
    tile_rows(rows, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    idma_memcpy_1d(0,
                   (uint32_t)(O + row_start * cols),
                   (uint32_t)rd_params->shard_O,
                   row_len * cols * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    /* This tile's rows of each of the n slices, packed into an n x row_len x cols block. */
    if (n > 0) {
        idma_memcpy_2d(0,
                       (uint32_t)(G + row_start * cols),
                       (uint32_t)rd_params->shard_G,
                       row_len * cols * sizeof(float16),
                       rows * cols * sizeof(float16),
                       n);
        eu_idma_wait_a2o(WFE);
    }

    idma_memcpy_1d(
        0, (uint32_t)(S + row_start), (uint32_t)rd_params->shard_S, row_len * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    spatz_run_task_with_params(ROWSLICE_SUM_DIV_FP16_SPATZ_TASK, (uint32_t)params);

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

void MAGIA_rowslice_sum_div_fp16_spatz_l1(float16 *Y,
                                          const float16 *O,
                                          const float16 *G,
                                          const float16 *S,
                                          uint32_t n,
                                          uint32_t own,
                                          uint32_t rows,
                                          uint32_t cols)
{
    volatile rowslice_sum_div_fp16_spatz_params_t params = {
        .shard_Y = (uintptr_t)Y,
        .shard_O = (uintptr_t)O,
        .shard_G = (uintptr_t)G,
        .shard_S = (uintptr_t)S,
        .dims    = rows | (cols << 16),
        .slices  = n | (own << 16),
    };
    int ret;

    ret = offload_spatz_task((void *)&params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
    }
}

static int store_result(void *params, float16 *Y, uint32_t rows)
{
    volatile rowslice_sum_div_fp16_spatz_params_t *rd_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    rd_params = (volatile rowslice_sum_div_fp16_spatz_params_t *)params;
    cols      = rd_params->dims >> 16;
    tile_rows(rows, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    idma_memcpy_1d(1,
                   (uint32_t)(Y + row_start * cols),
                   (uint32_t)rd_params->shard_Y,
                   row_len * cols * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    return 0;
}

void MAGIA_rowslice_sum_div_fp16_spatz(float16 *Y,
                                       const float16 *O,
                                       const float16 *G,
                                       const float16 *S,
                                       uint32_t n,
                                       uint32_t own,
                                       uint32_t rows,
                                       uint32_t cols)
{
    int ret;
    volatile rowslice_sum_div_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, n, own, rows, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, O, G, S, rows);
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

    ret = store_result((void *)params, Y, rows);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
