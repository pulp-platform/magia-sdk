#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "bias_rope_fp16_spatz.h"
#include "bias_rope_fp16_spatz_params.h"
#include "bias_rope_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "bias_rope_fp16_spatz"

/* This tile's rows [*row_start, *row_start + *row_len) out of rows. */
static void tile_rows(uint32_t rows, uint32_t *row_start, uint32_t *row_len)
{
    uint32_t elems = rows / NUM_HARTS;
    uint32_t left  = rows % NUM_HARTS;

    *row_start = HID * elems + (HID < left ? HID : left);
    *row_len   = elems + (HID < left ? 1 : 0);
}

static int alloc_l1(void **params,
                    const float16 *C,
                    uint32_t rows,
                    uint32_t cols,
                    uint32_t col0,
                    uint32_t head_dim,
                    uint32_t group)
{
    volatile bias_rope_fp16_spatz_params_t *br_params;
    uintptr_t shard_X;
    uintptr_t bias;
    uintptr_t cos = 0;
    uintptr_t sin = 0;
    uint32_t row_start;
    uint32_t row_len;

    tile_rows(rows, &row_start, &row_len);

    l1_alloc_init();

    br_params = l1_alloc(sizeof(bias_rope_fp16_spatz_params_t));
    if (!br_params)
        return ENOMEM;

    shard_X = (uintptr_t)l1_alloc(row_len * cols * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    bias = (uintptr_t)l1_alloc(cols * sizeof(float16));
    if (!bias)
        return ENOMEM;

    if (C) {
        cos = (uintptr_t)l1_alloc(row_len * (head_dim / 2) * sizeof(float16));
        if (!cos)
            return ENOMEM;

        sin = (uintptr_t)l1_alloc(row_len * (head_dim / 2) * sizeof(float16));
        if (!sin)
            return ENOMEM;
    }

    br_params->shard_X  = shard_X;
    br_params->bias     = bias;
    br_params->cos      = cos;
    br_params->sin      = sin;
    br_params->dims     = row_len | (cols << 16);
    br_params->col0     = col0;
    br_params->head_dim = head_dim;
    br_params->group    = group;

    *params = (void *)br_params;

    return 0;
}

static int init_input_params(void *params,
                             const float16 *X,
                             const float16 *B,
                             const float16 *C,
                             const float16 *S,
                             uint32_t rows)
{
    volatile bias_rope_fp16_spatz_params_t *br_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;
    uint32_t half;

    br_params = (volatile bias_rope_fp16_spatz_params_t *)params;
    cols      = br_params->dims >> 16;
    half      = br_params->head_dim / 2;
    tile_rows(rows, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    /*
     * This tile's rows (tokens) are contiguous in X, and so are their angles in C and S. Every
     * tile needs the whole bias.
     */
    idma_memcpy_1d(0,
                   (uint32_t)(X + row_start * cols),
                   (uint32_t)br_params->shard_X,
                   row_len * cols * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    idma_memcpy_1d(0, (uint32_t)B, (uint32_t)br_params->bias, cols * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    if (C) {
        idma_memcpy_1d(0,
                       (uint32_t)(C + row_start * half),
                       (uint32_t)br_params->cos,
                       row_len * half * sizeof(float16));
        eu_idma_wait_a2o(WFE);

        idma_memcpy_1d(0,
                       (uint32_t)(S + row_start * half),
                       (uint32_t)br_params->sin,
                       row_len * half * sizeof(float16));
        eu_idma_wait_a2o(WFE);
    }

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    spatz_run_task_with_params(BIAS_ROPE_FP16_SPATZ_TASK, (uint32_t)params);

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

void MAGIA_bias_rope_fp16_spatz_l1(float16 *X,
                                   const float16 *B,
                                   const float16 *C,
                                   const float16 *S,
                                   uint32_t rows,
                                   uint32_t cols,
                                   uint32_t col0,
                                   uint32_t head_dim,
                                   uint32_t group)
{
    volatile bias_rope_fp16_spatz_params_t params = {
        .shard_X  = (uintptr_t)X,
        .bias     = (uintptr_t)B,
        .cos      = (uintptr_t)C,
        .sin      = (uintptr_t)S,
        .dims     = rows | (cols << 16),
        .col0     = col0,
        .head_dim = head_dim,
        .group    = group,
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

static int store_result(void *params, float16 *X, uint32_t rows)
{
    volatile bias_rope_fp16_spatz_params_t *br_params;
    uint32_t row_start;
    uint32_t row_len;
    uint32_t cols;

    br_params = (volatile bias_rope_fp16_spatz_params_t *)params;
    cols      = br_params->dims >> 16;
    tile_rows(rows, &row_start, &row_len);

    if (row_len == 0)
        return 0;

    idma_memcpy_1d(1,
                   (uint32_t)(X + row_start * cols),
                   (uint32_t)br_params->shard_X,
                   row_len * cols * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    return 0;
}

void MAGIA_bias_rope_fp16_spatz(float16 *X,
                                const float16 *B,
                                const float16 *C,
                                const float16 *S,
                                uint32_t rows,
                                uint32_t cols,
                                uint32_t col0,
                                uint32_t head_dim,
                                uint32_t group)
{
    int ret;
    volatile bias_rope_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, C, rows, cols, col0, head_dim, group);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X, B, C, S, rows);
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
