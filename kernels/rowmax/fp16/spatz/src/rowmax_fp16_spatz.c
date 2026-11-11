#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "rowmax_fp16_spatz.h"
#include "rowmax_fp16_spatz_params.h"
#include "rowmax_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "rowmax_fp16_spatz"

/* This tile's rows [*row_start, *row_start + *row_len) out of rows. */
static void tile_rows(uint32_t rows, uint32_t *row_start, uint32_t *row_len)
{
    uint32_t elems = rows / NUM_HARTS;
    uint32_t left  = rows % NUM_HARTS;

    *row_start = HID * elems + (HID < left ? HID : left);
    *row_len   = elems + (HID < left ? 1 : 0);
}

static int alloc_l1(void **params, const float16 *P, uint32_t rows, uint32_t cols)
{
    volatile rowmax_fp16_spatz_params_t *rm_params;
    uintptr_t shard_X;
    uintptr_t shard_P = 0;
    uintptr_t shard_M;
    uint32_t row_start;
    uint32_t row_len;

    tile_rows(rows, &row_start, &row_len);

    l1_alloc_init();

    rm_params = l1_alloc(sizeof(rowmax_fp16_spatz_params_t));
    if (!rm_params)
        return ENOMEM;

    shard_X = (uintptr_t)l1_alloc(row_len * cols * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    if (P) {
        shard_P = (uintptr_t)l1_alloc(row_len * sizeof(float16));
        if (!shard_P)
            return ENOMEM;
    }

    shard_M = (uintptr_t)l1_alloc(row_len * sizeof(float16));
    if (!shard_M)
        return ENOMEM;

    rm_params->shard_X = shard_X;
    rm_params->shard_P = shard_P;
    rm_params->shard_M = shard_M;
    rm_params->dims    = row_len | (cols << 16);

    *params = (void *)rm_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, const float16 *P, uint32_t rows)
{
    volatile rowmax_fp16_spatz_params_t *rm_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    rm_params = (volatile rowmax_fp16_spatz_params_t *)params;
    cols      = rm_params->dims >> 16;
    tile_rows(rows, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    /* This tile's rows are contiguous in X, and so are their previous maxima in P. */
    idma_memcpy_1d(0,
                   (uint32_t)(X + row_start * cols),
                   (uint32_t)rm_params->shard_X,
                   row_len * cols * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    if (P) {
        idma_memcpy_1d(
            0, (uint32_t)(P + row_start), (uint32_t)rm_params->shard_P, row_len * sizeof(float16));
        eu_idma_wait_a2o(WFE);
    }

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    spatz_run_task_with_params(ROWMAX_FP16_SPATZ_TASK, (uint32_t)params);

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

void MAGIA_rowmax_fp16_spatz_l1(
    const float16 *X, const float16 *P, float16 *M, uint32_t rows, uint32_t cols)
{
    volatile rowmax_fp16_spatz_params_t params = {
        .shard_X = (uintptr_t)X,
        .shard_P = (uintptr_t)P,
        .shard_M = (uintptr_t)M,
        .dims    = rows | (cols << 16),
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

static int store_result(void *params, float16 *M, uint32_t rows)
{
    volatile rowmax_fp16_spatz_params_t *rm_params;
    uint32_t row_start;
    uint32_t row_len;

    rm_params = (volatile rowmax_fp16_spatz_params_t *)params;
    tile_rows(rows, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    idma_memcpy_1d(
        1, (uint32_t)(M + row_start), (uint32_t)rm_params->shard_M, row_len * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    return 0;
}

void MAGIA_rowmax_fp16_spatz(
    const float16 *X, const float16 *P, float16 *M, uint32_t rows, uint32_t cols)
{
    int ret;
    volatile rowmax_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, P, rows, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X, P, rows);
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

    ret = store_result((void *)params, M, rows);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
