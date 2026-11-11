#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "colmax_fp16_spatz.h"
#include "colmax_fp16_spatz_params.h"
#include "colmax_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "colmax_fp16_spatz"

/*
 * This tile's columns [*col_start, *col_start + *col_len) out of cols, split in pairs so that
 * each tile gets an even number of columns; an odd last column goes to the last tile.
 */
static void tile_cols(uint32_t cols, uint32_t *col_start, uint32_t *col_len)
{
    uint32_t pairs = cols / 2;
    uint32_t elems = pairs / NUM_HARTS;
    uint32_t left  = pairs % NUM_HARTS;

    *col_start = 2 * (HID * elems + (HID < left ? HID : left));
    *col_len   = 2 * (elems + (HID < left ? 1 : 0));
    if (HID == NUM_HARTS - 1)
        *col_len += cols & 1;
}

static int alloc_l1(void **params, uint32_t rows, uint32_t cols)
{
    volatile colmax_fp16_spatz_params_t *cm_params;
    uintptr_t shard_X;
    uintptr_t shard_M;
    uint32_t col_start;
    uint32_t col_len;

    tile_cols(cols, &col_start, &col_len);

    l1_alloc_init();

    cm_params = l1_alloc(sizeof(colmax_fp16_spatz_params_t));
    if (!cm_params)
        return ENOMEM;

    shard_X = (uintptr_t)l1_alloc(rows * col_len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_M = (uintptr_t)l1_alloc(col_len * sizeof(float16));
    if (!shard_M)
        return ENOMEM;

    cm_params->shard_X = shard_X;
    cm_params->shard_M = shard_M;
    cm_params->dims    = rows | (col_len << 16);

    *params = (void *)cm_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, uint32_t cols)
{
    volatile colmax_fp16_spatz_params_t *cm_params;
    uint32_t col_start;
    uint32_t col_len;
    uint32_t rows;

    cm_params = (volatile colmax_fp16_spatz_params_t *)params;
    rows      = cm_params->dims & 0xffff;
    tile_cols(cols, &col_start, &col_len);

    if (col_len == 0 || rows == 0)
        return 0;

    /* This tile's columns of every row, packed into a rows x col_len matrix. */
    idma_memcpy_2d(0,
                   (uint32_t)(X + col_start),
                   (uint32_t)cm_params->shard_X,
                   col_len * sizeof(float16),
                   cols * sizeof(float16),
                   rows);
    eu_idma_wait_a2o(WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    spatz_run_task_with_params(COLMAX_FP16_SPATZ_TASK, (uint32_t)params);

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

void MAGIA_colmax_fp16_spatz_l1(const float16 *X, float16 *M, uint32_t rows, uint32_t cols)
{
    volatile colmax_fp16_spatz_params_t params;
    int ret;

    /* Field by field: GCC hits an internal error on this 3-word volatile initializer. */
    params.shard_X = (uintptr_t)X;
    params.shard_M = (uintptr_t)M;
    params.dims    = rows | (cols << 16);

    ret = offload_spatz_task((void *)&params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
    }
}

static int store_result(void *params, float16 *M, uint32_t cols)
{
    volatile colmax_fp16_spatz_params_t *cm_params;
    uint32_t col_start;
    uint32_t col_len;

    cm_params = (volatile colmax_fp16_spatz_params_t *)params;
    tile_cols(cols, &col_start, &col_len);

    if (col_len == 0)
        return 0;

    idma_memcpy_1d(
        1, (uint32_t)(M + col_start), (uint32_t)cm_params->shard_M, col_len * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    return 0;
}

void MAGIA_colmax_fp16_spatz(const float16 *X, float16 *M, uint32_t rows, uint32_t cols)
{
    int ret;
    volatile colmax_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, rows, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X, cols);
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

    ret = store_result((void *)params, M, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
