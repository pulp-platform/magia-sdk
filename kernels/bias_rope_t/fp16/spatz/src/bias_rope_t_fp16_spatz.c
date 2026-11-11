#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "bias_rope_t_fp16_spatz.h"
#include "bias_rope_t_fp16_spatz_params.h"
#include "bias_rope_t_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "bias_rope_t_fp16_spatz"

/* This tile's columns [*col_start, *col_start + *col_len) out of cols, split in pairs so that
   each tile gets an even number of columns; an odd last column goes to the last tile. */
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

static int alloc_l1(
    void **params, uint32_t rows, uint32_t cols, uint32_t row0, uint32_t head_dim, uint32_t group)
{
    volatile bias_rope_t_fp16_spatz_params_t *br_params;
    uintptr_t shard_X;
    uintptr_t bias;
    uintptr_t cos;
    uintptr_t sin;
    uint32_t col_start;
    uint32_t col_len;

    tile_cols(cols, &col_start, &col_len);

    l1_alloc_init();

    br_params = l1_alloc(sizeof(bias_rope_t_fp16_spatz_params_t));
    if (!br_params)
        return ENOMEM;

    shard_X = (uintptr_t)l1_alloc(rows * col_len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    bias = (uintptr_t)l1_alloc(rows * sizeof(float16));
    if (!bias)
        return ENOMEM;

    cos = (uintptr_t)l1_alloc((head_dim / 2) * col_len * sizeof(float16));
    if (!cos)
        return ENOMEM;

    sin = (uintptr_t)l1_alloc((head_dim / 2) * col_len * sizeof(float16));
    if (!sin)
        return ENOMEM;

    br_params->shard_X  = shard_X;
    br_params->bias     = bias;
    br_params->cos      = cos;
    br_params->sin      = sin;
    br_params->dims     = rows | (col_len << 16);
    br_params->row0     = row0;
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
                             uint32_t cols)
{
    volatile bias_rope_t_fp16_spatz_params_t *br_params;
    uint32_t col_start;
    uint32_t col_len;
    uint32_t rows;
    uint32_t half;

    br_params = (volatile bias_rope_t_fp16_spatz_params_t *)params;
    rows      = br_params->dims & 0xffff;
    half      = br_params->head_dim / 2;
    tile_cols(cols, &col_start, &col_len);

    if (col_len == 0)
        return 0;

    /* This tile's columns (tokens) of every row of X, C and S, packed into matrices of col_len
       columns; every tile needs the whole bias. */
    if (rows > 0) {
        idma_memcpy_2d(0,
                       (uint32_t)(X + col_start),
                       (uint32_t)br_params->shard_X,
                       col_len * sizeof(float16),
                       cols * sizeof(float16),
                       rows);
        eu_idma_wait_a2o(WFE);

        idma_memcpy_1d(0, (uint32_t)B, (uint32_t)br_params->bias, rows * sizeof(float16));
        eu_idma_wait_a2o(WFE);
    }

    if (half > 0) {
        idma_memcpy_2d(0,
                       (uint32_t)(C + col_start),
                       (uint32_t)br_params->cos,
                       col_len * sizeof(float16),
                       cols * sizeof(float16),
                       half);
        eu_idma_wait_a2o(WFE);

        idma_memcpy_2d(0,
                       (uint32_t)(S + col_start),
                       (uint32_t)br_params->sin,
                       col_len * sizeof(float16),
                       cols * sizeof(float16),
                       half);
        eu_idma_wait_a2o(WFE);
    }

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    spatz_run_task_with_params(BIAS_ROPE_T_FP16_SPATZ_TASK, (uint32_t)params);

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

void MAGIA_bias_rope_t_fp16_spatz_l1(float16 *X,
                                     const float16 *B,
                                     const float16 *C,
                                     const float16 *S,
                                     uint32_t rows,
                                     uint32_t cols,
                                     uint32_t row0,
                                     uint32_t head_dim,
                                     uint32_t group)
{
    volatile bias_rope_t_fp16_spatz_params_t params = {
        .shard_X  = (uintptr_t)X,
        .bias     = (uintptr_t)B,
        .cos      = (uintptr_t)C,
        .sin      = (uintptr_t)S,
        .dims     = rows | (cols << 16),
        .row0     = row0,
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

static int store_result(void *params, float16 *X, uint32_t cols)
{
    volatile bias_rope_t_fp16_spatz_params_t *br_params;
    uint32_t col_start;
    uint32_t col_len;
    uint32_t rows;

    br_params = (volatile bias_rope_t_fp16_spatz_params_t *)params;
    rows      = br_params->dims & 0xffff;
    tile_cols(cols, &col_start, &col_len);

    if (col_len == 0 || rows == 0)
        return 0;

    idma_memcpy_2d(1,
                   (uint32_t)(X + col_start),
                   (uint32_t)br_params->shard_X,
                   col_len * sizeof(float16),
                   cols * sizeof(float16),
                   rows);
    eu_idma_wait_o2a(WFE);

    return 0;
}

void MAGIA_bias_rope_t_fp16_spatz(float16 *X,
                                  const float16 *B,
                                  const float16 *C,
                                  const float16 *S,
                                  uint32_t rows,
                                  uint32_t cols,
                                  uint32_t row0,
                                  uint32_t head_dim,
                                  uint32_t group)
{
    int ret;
    volatile bias_rope_t_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, rows, cols, row0, head_dim, group);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X, B, C, S, cols);
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

    ret = store_result((void *)params, X, cols);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
