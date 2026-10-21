#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "expminus_fp16_spatz.h"
#include "expminus_fp16_spatz_params.h"
#include "expminus_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "expminus_fp16_spatz"

static int alloc_l1(void **params, uint32_t rows, uint32_t cols)
{
    volatile expminus_fp16_spatz_params_t *em_params;
    uintptr_t shard_X;
    uintptr_t shard_M;
    uintptr_t shard_Y;

    size_t row_start;
    size_t row_end;
    size_t row_len;
    size_t elems;
    size_t left;

    elems = rows / NUM_HARTS;
    left  = rows % NUM_HARTS;

    row_start = HID * elems + (HID < left ? HID : left);
    row_end   = row_start + elems + (HID < left ? 1 : 0);
    row_len   = row_end - row_start;

    l1_alloc_init();

    em_params = l1_alloc(sizeof(expminus_fp16_spatz_params_t));
    if (!em_params)
        return ENOMEM;

    shard_X = l1_alloc(row_len * cols * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_M = l1_alloc(row_len * sizeof(float16));
    if (!shard_M)
        return ENOMEM;

    shard_Y = l1_alloc(row_len * cols * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    em_params->shard_X   = shard_X;
    em_params->shard_M   = shard_M;
    em_params->shard_Y   = shard_Y;
    em_params->cols      = cols;
    em_params->row_start = row_start;
    em_params->row_len   = row_len;

    *params = (void *)em_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, const float16 *M)
{
    volatile expminus_fp16_spatz_params_t *em_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    em_params = (volatile expminus_fp16_spatz_params_t *)params;
    row_start = em_params->row_start;
    row_len   = em_params->row_len;
    cols      = em_params->cols;

    if (row_len == 0)
        return 0;

    /* This tile's rows [row_start, row_start+row_len) are contiguous in X, and so are their
       subtrahends in M. The Spatz task writes every output, so shard_Y is not zeroed here. */
    idma_memcpy_1d(0,
                   (uint32_t)(X + row_start * cols),
                   (uint32_t)em_params->shard_X,
                   row_len * cols * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    idma_memcpy_1d(
        0, (uint32_t)(M + row_start), (uint32_t)em_params->shard_M, row_len * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    spatz_run_task_with_params(EXPMINUS_FP16_SPATZ_TASK, (uint32_t)params);

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

/* Y[r, c] = exp(X[r, c] - M[r]) on this tile's Spatz, for operands that are already in this
   tile's L1: no sharding across tiles, no L1 allocation and no L2 transfers. Y may alias X. */
void MAGIA_expminus_fp16_spatz_l1(
    const float16 *X, const float16 *M, float16 *Y, uint32_t rows, uint32_t cols)
{
    volatile expminus_fp16_spatz_params_t params = {
        .shard_X   = (uintptr_t)X,
        .shard_M   = (uintptr_t)M,
        .shard_Y   = (uintptr_t)Y,
        .cols      = cols,
        .row_start = 0,
        .row_len   = rows,
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

static int store_result(void *params, float16 *Y)
{
    volatile expminus_fp16_spatz_params_t *em_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    em_params = (volatile expminus_fp16_spatz_params_t *)params;
    row_start = em_params->row_start;
    row_len   = em_params->row_len;
    cols      = em_params->cols;

    if (row_len == 0)
        return 0;

    /* This tile's output rows are contiguous in L2. */
    idma_memcpy_1d(1,
                   (uint32_t)(Y + row_start * cols),
                   (uint32_t)em_params->shard_Y,
                   row_len * cols * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    return 0;
}

void MAGIA_expminus_fp16_spatz(
    const float16 *X, const float16 *M, float16 *Y, uint32_t rows, uint32_t cols)
{
    int ret;
    volatile expminus_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, rows, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X, M);
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

    ret = store_result((void *)params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
