#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "batchnorm_fp16_spatz.h"
#include "batchnorm_fp16_spatz_params.h"
#include "batchnorm_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int alloc_l1(void **params, uint32_t shape[4])
{
    volatile batchnorm_fp16_spatz_params_t *batchnorm_params;
    uint32_t shard_X;
    uint32_t shard_Y;
    uintptr_t gamma;
    uintptr_t beta;
    uintptr_t mean;
    uintptr_t var;
    uintptr_t eps;

    size_t c_start;
    size_t c_end;
    size_t c_len;
    size_t elems;
    size_t left;

    size_t B;
    size_t C;
    size_t HW;
    size_t iterations;

    B = shape[0];
    C = shape[1];
    iterations = B * C;
    HW = shape[2] * shape[3];

    elems = iterations / NUM_HARTS;
    left = iterations % NUM_HARTS;

    c_start = HID * elems + (HID < left ? HID : left);
    c_end = c_start + elems + (HID < left ? 1 : 0);
    c_len = c_end - c_start;

    l1_alloc_init();

    batchnorm_params = l1_alloc(sizeof(batchnorm_fp16_spatz_params_t));
    if (!batchnorm_params)
        return ENOMEM;

    shard_X = l1_alloc(c_len * HW * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(c_len * HW * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    gamma = l1_alloc(C * sizeof(float16));
    if (!gamma)
        return ENOMEM;

    beta = l1_alloc(C * sizeof(float16));
    if (!beta)
        return ENOMEM;

    mean = l1_alloc(C * sizeof(float16));
    if (!mean)
        return ENOMEM;

    var = l1_alloc(C * sizeof(float16));
    if (!var)
        return ENOMEM;

    eps = l1_alloc(sizeof(float16));
    if (!eps)
        return ENOMEM;

    batchnorm_params->shard_X = shard_X;
    batchnorm_params->shard_Y = shard_Y;
    batchnorm_params->gamma = gamma;
    batchnorm_params->beta = beta;
    batchnorm_params->mean = mean;
    batchnorm_params->var = var;
    batchnorm_params->eps = eps;
    batchnorm_params->c_start = c_start;
    batchnorm_params->c_len = c_len;
    batchnorm_params->hw_len = HW;
    batchnorm_params->channels = C;

    *params = (void *) batchnorm_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, const float16 *scale, const float16 *B, const float16 *input_mean, const float16* input_var, const float16 epsilon)
{
    volatile batchnorm_fp16_spatz_params_t *batchnorm_params;
    uint32_t shard_X;
    uint32_t shard_Y;
    uintptr_t gamma;
    uintptr_t beta;
    uintptr_t mean;
    uintptr_t var;
    uintptr_t eps;

    size_t channels;
    size_t c_start;
    size_t c_end;
    size_t c_len;
    size_t hw;

    batchnorm_params = (volatile batchnorm_fp16_spatz_params_t *) params;

    shard_X = batchnorm_params->shard_X;
    shard_Y = batchnorm_params->shard_Y;
    gamma = batchnorm_params->gamma;
    beta = batchnorm_params->beta;
    mean = batchnorm_params->mean;
    var = batchnorm_params->var;
    eps = batchnorm_params->eps;

    c_start = batchnorm_params->c_start;
    c_len = batchnorm_params->c_len;
    c_end = c_start + c_len;
    hw = batchnorm_params->hw_len;
    channels = batchnorm_params->channels;

    uint32_t local_idx = 0;
    for (uint32_t c = c_start; c < c_end; c++) {
        uint32_t c_base;

        c_base = c * hw;

        for (uint32_t i = 0; i < hw; i++) {
            uint32_t global_idx;
            uint32_t offset;

            global_idx = c_base + i;
            offset = local_idx * sizeof(float16);

            mmio_fp16(shard_X + offset) = X[global_idx];
            mmio_fp16(shard_Y + offset) = 0;

            local_idx++;
        }
    }

    for (uint32_t i = 0; i < channels; i++) {
        uint32_t offset = i * sizeof(float16);

        mmio_fp16(mean + offset) = input_mean[i];
        mmio_fp16(var + offset) = input_var[i];
        mmio_fp16(gamma + offset) = scale[i];
        mmio_fp16(beta + offset) = B[i];
    }

    mmio_fp16(eps) = epsilon;

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    eu_config_t eu_cfg;
    int ret;

    eu_cfg.hartid = HID;
    eu_ctrl.base = NULL;
    eu_ctrl.cfg = &eu_cfg;
    eu_ctrl.api = &eu_api;

    spatz_init(SPATZ_BINARY_START);
    spatz_run_task_with_params(BATCHNORM_FP16_SPATZ_TASK, params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] Wait on Spatz task completion failed with error: %d\n", HID, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();
    spatz_clk_dis();

exit:
    return ret;
}

static int store_result(void *params, float16 *Y)
{
    volatile batchnorm_fp16_spatz_params_t *batchnorm_params;
    uint32_t shard_Y_base;
    uint32_t local_idx;

    batchnorm_params = (volatile batchnorm_fp16_spatz_params_t *) params;
    shard_Y_base = batchnorm_params->shard_Y;
    local_idx = 0;

    for (uint32_t c = 0; c < batchnorm_params->c_len; c++) {
        uint32_t c_base;
        uint32_t c_idx;

        c_idx = batchnorm_params->c_start + c;
        c_base = c_idx * batchnorm_params->hw_len;

        for (uint32_t i = 0; i < batchnorm_params->hw_len; i++) {
            uint32_t global_idx;
            uint32_t offset;

            global_idx = c_base + i;
            offset = local_idx * sizeof(float16);

            Y[global_idx] = mmio_fp16(shard_Y_base + offset);

            local_idx++;
        }
    }

    return 0;
}

void MAGIA_batchnorm_fp16_spatz(const float16 *X, const float16 *scale, const float16 *B, const float16 *input_mean, const float16* input_var, const float16 epsilon, float16 *Y, uint32_t shape[4])
{
    int ret;
    volatile batchnorm_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, shape);
    if (ret != 0) {
        printf("[CV32 (%d)] L1 allocation failed with error: %d\n", HID, ret);
        return;
    }

    ret = init_input_params(params, X, scale, B, input_mean, input_var, epsilon);
    if (ret != 0) {
        printf("[CV32 (%d)] Params initialization failed with error: %d\n", HID, ret);
        return;
    }

    ret = offload_spatz_task(params);
    if (ret != 0) {
        printf("[CV32 (%d)] Spatz task offloading failed with error: %d\n", HID, ret);
        return;
    }

    ret = store_result(params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] Result write back failed with error: %d\n", HID, ret);
    }

}
