#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "colsum_rescale_fp16_spatz.h"
#include "colsum_rescale_fp16_spatz_params.h"
#include "colsum_rescale_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "colsum_rescale_fp16_spatz"

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

static int alloc_l1(void **params, const float16 *P, uint32_t rows, uint32_t cols)
{
    volatile colsum_rescale_fp16_spatz_params_t *cs_params;
    uintptr_t shard_S;
    uintptr_t shard_G;
    uintptr_t shard_P = 0;
    uintptr_t shard_A = 0;
    uint32_t col_start;
    uint32_t col_len;

    tile_cols(cols, &col_start, &col_len);

    l1_alloc_init();

    cs_params = l1_alloc(sizeof(colsum_rescale_fp16_spatz_params_t));
    if (!cs_params)
        return ENOMEM;

    shard_S = (uintptr_t)l1_alloc(col_len * sizeof(float16));
    if (!shard_S)
        return ENOMEM;

    shard_G = (uintptr_t)l1_alloc(rows * col_len * sizeof(float16));
    if (!shard_G)
        return ENOMEM;

    if (P) {
        shard_P = (uintptr_t)l1_alloc(col_len * sizeof(float16));
        if (!shard_P)
            return ENOMEM;

        shard_A = (uintptr_t)l1_alloc(col_len * sizeof(float16));
        if (!shard_A)
            return ENOMEM;
    }

    cs_params->shard_S = shard_S;
    cs_params->shard_G = shard_G;
    cs_params->shard_P = shard_P;
    cs_params->shard_A = shard_A;
    cs_params->dims    = rows | (col_len << 16);

    *params = (void *)cs_params;

    return 0;
}

static int init_input_params(void *params,
                             const float16 *S,
                             const float16 *G,
                             const float16 *P,
                             const float16 *A,
                             uint32_t cols)
{
    volatile colsum_rescale_fp16_spatz_params_t *cs_params;
    uint32_t col_start;
    uint32_t col_len;
    uint32_t rows;

    cs_params = (volatile colsum_rescale_fp16_spatz_params_t *)params;
    rows      = cs_params->dims & 0xffff;
    tile_cols(cols, &col_start, &col_len);

    if (col_len == 0)
        return 0;

    idma_memcpy_1d(
        0, (uint32_t)(S + col_start), (uint32_t)cs_params->shard_S, col_len * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    /*
     * This tile's columns of every row of G, packed into a rows x col_len matrix (row 0 is
     * unused by the task, but keeps the row indexing).
     */
    if (rows > 0) {
        idma_memcpy_2d(0,
                       (uint32_t)(G + col_start),
                       (uint32_t)cs_params->shard_G,
                       col_len * sizeof(float16),
                       cols * sizeof(float16),
                       rows);
        eu_idma_wait_a2o(WFE);
    }

    if (P) {
        idma_memcpy_1d(
            0, (uint32_t)(P + col_start), (uint32_t)cs_params->shard_P, col_len * sizeof(float16));
        eu_idma_wait_a2o(WFE);

        idma_memcpy_1d(
            0, (uint32_t)(A + col_start), (uint32_t)cs_params->shard_A, col_len * sizeof(float16));
        eu_idma_wait_a2o(WFE);
    }

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    spatz_run_task_with_params(COLSUM_RESCALE_FP16_SPATZ_TASK, (uint32_t)params);

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

void MAGIA_colsum_rescale_fp16_spatz_l1(
    float16 *S, const float16 *G, float16 *P, const float16 *A, uint32_t rows, uint32_t cols)
{
    volatile colsum_rescale_fp16_spatz_params_t params = {
        .shard_S = (uintptr_t)S,
        .shard_G = (uintptr_t)G,
        .shard_P = (uintptr_t)P,
        .shard_A = (uintptr_t)A,
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

static int store_result(void *params, float16 *S, float16 *P, uint32_t cols)
{
    volatile colsum_rescale_fp16_spatz_params_t *cs_params;
    uint32_t col_start;
    uint32_t col_len;

    cs_params = (volatile colsum_rescale_fp16_spatz_params_t *)params;
    tile_cols(cols, &col_start, &col_len);

    if (col_len == 0)
        return 0;

    idma_memcpy_1d(
        1, (uint32_t)(S + col_start), (uint32_t)cs_params->shard_S, col_len * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    /* P is rescaled in place. */
    if (P) {
        idma_memcpy_1d(
            1, (uint32_t)(P + col_start), (uint32_t)cs_params->shard_P, col_len * sizeof(float16));
        eu_idma_wait_o2a(WFE);
    }

    return 0;
}

void MAGIA_colsum_rescale_fp16_spatz(
    float16 *S, const float16 *G, float16 *P, const float16 *A, uint32_t rows, uint32_t cols)
{
    int ret;
    volatile colsum_rescale_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, P, rows, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, S, G, P, A, cols);
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

    ret = store_result((void *)params, S, P, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
