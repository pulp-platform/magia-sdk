#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "rope_fp16_spatz.h"
#include "rope_fp16_spatz_params.h"
#include "rope_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "rope_fp16_spatz"

static int alloc_l1(void **params, uint32_t rows, uint32_t n_heads, uint32_t head_dim)
{
    volatile rope_fp16_spatz_params_t *rope_params;
    uint32_t shard_X;
    uint32_t cos;
    uint32_t sin;

    size_t r_start;
    size_t r_end;
    size_t r_len;
    size_t shard;
    size_t left;

    shard = rows / NUM_HARTS;
    left  = rows % NUM_HARTS;

    r_start = HID * shard + (HID < left ? HID : left);
    r_end   = r_start + shard + (HID < left ? 1 : 0);
    r_len   = r_end - r_start;

    l1_alloc_init();

    rope_params = l1_alloc(sizeof(rope_fp16_spatz_params_t));
    if (!rope_params)
        return ENOMEM;

    /* The task rotates the rows in place: one shard for X and Y. */
    shard_X = (uint32_t)l1_alloc(r_len * n_heads * head_dim * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    cos = (uint32_t)l1_alloc(r_len * (head_dim / 2) * sizeof(float16));
    if (!cos)
        return ENOMEM;

    sin = (uint32_t)l1_alloc(r_len * (head_dim / 2) * sizeof(float16));
    if (!sin)
        return ENOMEM;

    rope_params->shard_X  = shard_X;
    rope_params->shard_Y  = shard_X;
    rope_params->cos      = cos;
    rope_params->sin      = sin;
    rope_params->r_start  = r_start;
    rope_params->r_len    = r_len;
    rope_params->n_heads  = n_heads;
    rope_params->head_dim = head_dim;

    *params = (void *)rope_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, const float16 *cos, const float16 *sin)
{
    volatile rope_fp16_spatz_params_t *rope_params;
    uint32_t r_start;
    uint32_t r_len;
    uint32_t width;
    uint32_t half;

    rope_params = (volatile rope_fp16_spatz_params_t *)params;
    r_start     = rope_params->r_start;
    r_len       = rope_params->r_len;
    width       = rope_params->n_heads * rope_params->head_dim;
    half        = rope_params->head_dim / 2;

    if (r_len == 0)
        return 0;

    /* This tile's rows [r_start, r_start+r_len) are contiguous in X, cos and sin. */
    idma_memcpy_1d(0,
                   (uint32_t)(X + r_start * width),
                   (uint32_t)rope_params->shard_X,
                   r_len * width * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    idma_memcpy_1d(0,
                   (uint32_t)(cos + r_start * half),
                   (uint32_t)rope_params->cos,
                   r_len * half * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    idma_memcpy_1d(0,
                   (uint32_t)(sin + r_start * half),
                   (uint32_t)rope_params->sin,
                   r_len * half * sizeof(float16));
    eu_idma_wait_a2o(WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    int ret;

    spatz_run_task_with_params(ROPE_FP16_SPATZ_TASK, (uint32_t)params);

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

/* RoPE of rows already in this tile's L1: no sharding across tiles, no L1 allocation and no L2
   transfers. Y may alias X. */
void MAGIA_rope_fp16_spatz_l1(const float16 *X,
                              const float16 *cos,
                              const float16 *sin,
                              uint32_t rows,
                              uint32_t n_heads,
                              uint32_t head_dim,
                              float16 *Y)
{
    volatile rope_fp16_spatz_params_t params = {
        .shard_X  = (uintptr_t)X,
        .shard_Y  = (uintptr_t)Y,
        .cos      = (uintptr_t)cos,
        .sin      = (uintptr_t)sin,
        .r_start  = 0,
        .r_len    = rows,
        .n_heads  = n_heads,
        .head_dim = head_dim,
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
    volatile rope_fp16_spatz_params_t *rope_params;
    uint32_t r_start;
    uint32_t r_len;
    uint32_t width;

    rope_params = (volatile rope_fp16_spatz_params_t *)params;
    r_start     = rope_params->r_start;
    r_len       = rope_params->r_len;
    width       = rope_params->n_heads * rope_params->head_dim;

    if (r_len == 0)
        return 0;

    /* This tile's output rows are contiguous in Y. */
    idma_memcpy_1d(1,
                   (uint32_t)(Y + r_start * width),
                   (uint32_t)rope_params->shard_Y,
                   r_len * width * sizeof(float16));
    eu_idma_wait_o2a(WFE);

    return 0;
}

void MAGIA_rope_fp16_spatz(const float16 *X,
                           const float16 *cos,
                           const float16 *sin,
                           uint32_t rows,
                           uint32_t n_heads,
                           uint32_t head_dim,
                           float16 *Y)
{
    int ret;
    volatile rope_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, rows, n_heads, head_dim);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X, cos, sin);
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
