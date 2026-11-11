#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "expminus_rowsum_fp16_spatz.h"
#include "expminus_rowsum_fp16_spatz_params.h"
#include "expminus_rowsum_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "expminus_rowsum_fp16_spatz"

/* This tile's rows [*row_start, *row_start + *row_len) out of rows. */
static void tile_rows(uint32_t rows, uint32_t *row_start, uint32_t *row_len)
{
    uint32_t elems = rows / NUM_HARTS;
    uint32_t left  = rows % NUM_HARTS;

    *row_start = HID * elems + (HID < left ? HID : left);
    *row_len   = elems + (HID < left ? 1 : 0);
}

static int alloc_l1(void **params, const float16 *A, float16 s, uint32_t rows, uint32_t cols)
{
    expminus_rowsum_fp16_spatz_params_t *er_params;
    uintptr_t shard_X;
    uintptr_t shard_M;
    uintptr_t shard_S;
    uintptr_t shard_A = 0;
    uint32_t row_start;
    uint32_t row_len;

    tile_rows(rows, &row_start, &row_len);

    l1_alloc_init();

    er_params = l1_alloc(sizeof(expminus_rowsum_fp16_spatz_params_t));
    if (!er_params)
        return ENOMEM;

    /* The task may write Y over X: one shard for X and Y. */
    shard_X = (uintptr_t)l1_alloc(row_len * cols * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_M = (uintptr_t)l1_alloc(row_len * sizeof(float16));
    if (!shard_M)
        return ENOMEM;

    shard_S = (uintptr_t)l1_alloc(row_len * sizeof(float16));
    if (!shard_S)
        return ENOMEM;

    if (A) {
        shard_A = (uintptr_t)l1_alloc(row_len * sizeof(float16));
        if (!shard_A)
            return ENOMEM;
    }

    er_params->shard_X = shard_X;
    er_params->shard_M = shard_M;
    er_params->shard_Y = shard_X;
    er_params->shard_S = shard_S;
    er_params->shard_A = shard_A;
    expminus_rowsum_fp16_spatz_set_scale(er_params, s, row_len, cols);

    *params = (void *)er_params;

    return 0;
}

static int
init_input_params(void *params, const float16 *X, const float16 *M, const float16 *A, uint32_t rows)
{
    expminus_rowsum_fp16_spatz_params_t *er_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    er_params = (expminus_rowsum_fp16_spatz_params_t *)params;
    cols      = er_params->dims >> 16;
    tile_rows(rows, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    /* This tile's rows are contiguous in X, and so are their entries in M and A. */
    idma_memcpy_1d(0,
                   (uint32_t)(X + row_start * cols),
                   (uint32_t)er_params->shard_X,
                   row_len * cols * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    idma_memcpy_1d(
        0, (uint32_t)(M + row_start), (uint32_t)er_params->shard_M, row_len * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    if (A) {
        idma_memcpy_1d(
            0, (uint32_t)(A + row_start), (uint32_t)er_params->shard_A, row_len * sizeof(float16));
        eu_idma_wait_a2o(WFE);
    }

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    /* The parameter stores must land before Spatz is triggered. */
    asm volatile("" ::: "memory");

    spatz_run_task_with_params(EXPMINUS_ROWSUM_FP16_SPATZ_TASK, (uint32_t)params);

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

void MAGIA_expminus_rowsum_fp16_spatz_l1(const float16 *X,
                                         const float16 *M,
                                         float16 *Y,
                                         float16 *S,
                                         float16 *A,
                                         float16 s,
                                         uint32_t rows,
                                         uint32_t cols)
{
    expminus_rowsum_fp16_spatz_params_t params = {
        .shard_X = (uintptr_t)X,
        .shard_M = (uintptr_t)M,
        .shard_Y = (uintptr_t)Y,
        .shard_S = (uintptr_t)S,
        .shard_A = (uintptr_t)A,
    };
    int ret;

    expminus_rowsum_fp16_spatz_set_scale(&params, s, rows, cols);

    ret = offload_spatz_task((void *)&params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
    }
}

static int store_result(void *params, float16 *Y, float16 *S, float16 *A, uint32_t rows)
{
    expminus_rowsum_fp16_spatz_params_t *er_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    er_params = (expminus_rowsum_fp16_spatz_params_t *)params;
    cols      = er_params->dims >> 16;
    tile_rows(rows, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    idma_memcpy_1d(1,
                   (uint32_t)(Y + row_start * cols),
                   (uint32_t)er_params->shard_Y,
                   row_len * cols * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    idma_memcpy_1d(
        1, (uint32_t)(S + row_start), (uint32_t)er_params->shard_S, row_len * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    /* A is updated in place. */
    if (A) {
        idma_memcpy_1d(
            1, (uint32_t)(A + row_start), (uint32_t)er_params->shard_A, row_len * sizeof(float16));
        eu_idma_wait_o2a(WFE);
    }

    return 0;
}

void MAGIA_expminus_rowsum_fp16_spatz(const float16 *X,
                                      const float16 *M,
                                      float16 *Y,
                                      float16 *S,
                                      float16 *A,
                                      float16 s,
                                      uint32_t rows,
                                      uint32_t cols)
{
    int ret;
    expminus_rowsum_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, A, s, rows, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X, M, A, rows);
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

    ret = store_result((void *)params, Y, S, A, rows);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
