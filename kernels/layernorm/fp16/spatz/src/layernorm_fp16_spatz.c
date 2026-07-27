#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"

#include "layernorm_fp16_spatz.h"
#include "layernorm_fp16_spatz_params.h"
#include "layernorm_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "layernorm_fp16_spatz"

static int alloc_l1(void **params, uint32_t *input_shape, uint32_t rank, float16 epsilon)
{
    volatile layernorm_fp16_spatz_params_t *layernorm_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;
    uintptr_t gamma;
    uintptr_t beta;
    uintptr_t eps;

    size_t total_rows;
    size_t shard;
    size_t left;
    size_t r_start;
    size_t r_end;
    size_t r_len;
    size_t w_len;
    size_t local_shard_size;

    /* LayerNorm normalizes over the last axis (axis == -1): the last dimension
       is the row length, the product of the leading dimensions is the number of
       rows. This is rank-agnostic (any tensor rank). */
    w_len = input_shape[rank - 1];
    total_rows = 1;
    for (uint32_t d = 0; d < rank - 1; d++)
        total_rows *= input_shape[d];

    shard = total_rows / NUM_HARTS;
    left = total_rows % NUM_HARTS;

    r_start = HID * shard + (HID < left ? HID : left);
    r_end = r_start + shard + (HID < left ? 1 : 0);
    r_len = r_end - r_start;

    local_shard_size = r_len * w_len;

    l1_alloc_init();

    layernorm_params = l1_alloc(sizeof(layernorm_fp16_spatz_params_t));
    if (!layernorm_params)
        return ENOMEM;

    shard_X = l1_alloc(local_shard_size * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(local_shard_size * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    gamma = l1_alloc(w_len * sizeof(float16));
    if (!gamma)
        return ENOMEM;

    beta = l1_alloc(w_len * sizeof(float16));
    if (!beta)
        return ENOMEM;

    eps = l1_alloc(sizeof(float16));
    if (!eps)
        return ENOMEM;

    layernorm_params->shard_X = shard_X;
    layernorm_params->shard_Y = shard_Y;
    layernorm_params->gamma   = gamma;
    layernorm_params->beta    = beta;
    layernorm_params->eps     = eps;
    layernorm_params->r_start = (uint32_t) r_start;
    layernorm_params->r_len   = (uint32_t) r_len;
    layernorm_params->w_len   = (uint32_t) w_len;

    *params = (void *) layernorm_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, const float16 *scale, const float16 *B, const float16 epsilon)
{
    volatile layernorm_fp16_spatz_params_t *layernorm_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t r_start;
    uint32_t w_len;
    uint32_t rows_bytes;

    layernorm_params = (volatile layernorm_fp16_spatz_params_t *) params;
    r_start = layernorm_params->r_start;
    w_len   = layernorm_params->w_len;
    rows_bytes = layernorm_params->r_len * w_len * sizeof(float16);

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's rows [r_start, r_start+r_len) are contiguous in L2, so X, gamma,
       beta and the output shard are plain 1D transfers. The Spatz task overwrites
       the whole output shard, so it does not need zeroing here. */
    if (rows_bytes > 0) {
        idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (X + r_start * w_len), (uint32_t) layernorm_params->shard_X, rows_bytes);
        eu_idma_wait_a2o(&eu_ctrl, WFE);
    }

    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) scale, (uint32_t) layernorm_params->gamma, w_len * sizeof(float16));
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) B, (uint32_t) layernorm_params->beta, w_len * sizeof(float16));
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    mmio_fp16(layernorm_params->eps) = epsilon;

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    int ret;

    eu_ctrl_init(&eu_ctrl);

    spatz_run_task_with_params(LAYERNORM_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *Y)
{
    volatile layernorm_fp16_spatz_params_t *layernorm_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t r_start;
    uint32_t w_len;
    uint32_t rows_bytes;

    layernorm_params = (volatile layernorm_fp16_spatz_params_t *) params;
    r_start = layernorm_params->r_start;
    w_len   = layernorm_params->w_len;
    rows_bytes = layernorm_params->r_len * w_len * sizeof(float16);

    if (rows_bytes == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's output rows are contiguous in L2 -> single 1D transfer (L1->L2). */
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t) (Y + r_start * w_len), (uint32_t) layernorm_params->shard_Y, rows_bytes);
    eu_idma_wait_o2a(&eu_ctrl, WFE);

    return 0;
}

void MAGIA_layernorm_fp16_spatz(const float16 *X, const float16 *scale, const float16 *B, const float16 epsilon, uint32_t *input_shape, uint32_t rank, float16 *Y)
{
    int ret;
    volatile layernorm_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, input_shape, rank, epsilon);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params(params, X, scale, B, epsilon);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task(params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result(params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
