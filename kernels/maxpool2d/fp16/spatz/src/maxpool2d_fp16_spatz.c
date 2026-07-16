#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "maxpool2d_fp16_spatz.h"
#include "maxpool2d_fp16_spatz_params.h"
#include "maxpool2d_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int alloc_l1(void **params, uint32_t input_shape[4], uint32_t output_shape[4])
{
    volatile maxpool2d_fp16_spatz_params_t *maxpool_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;

    size_t total_channels;
    size_t shard;
    size_t left;
    size_t c_start;
    size_t c_end;
    size_t c_len;

    size_t in_hw_len;
    size_t out_hw_len;
    size_t local_in_size;
    size_t local_out_size;

    total_channels = input_shape[0] * input_shape[1];
    shard = total_channels / NUM_HARTS;
    left  = total_channels % NUM_HARTS;

    c_start = HID * shard + (HID < left ? HID : left);
    c_end   = c_start + shard + (HID < left ? 1 : 0);
    c_len   = c_end - c_start;

    in_hw_len  = input_shape[2] * input_shape[3];
    out_hw_len = output_shape[2] * output_shape[3];

    local_in_size  = c_len * in_hw_len;
    local_out_size = c_len * out_hw_len;

    l1_alloc_init();

    maxpool_params = l1_alloc(sizeof(maxpool2d_fp16_spatz_params_t));
    if (!maxpool_params)
        return ENOMEM;

    shard_X = l1_alloc(local_in_size * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(local_out_size * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    maxpool_params->shard_X  = shard_X;
    maxpool_params->shard_Y  = shard_Y;
    maxpool_params->c_start  = (uint32_t) c_start;
    maxpool_params->c_len    = (uint32_t) c_len;
    maxpool_params->h_in     = input_shape[2];
    maxpool_params->w_in     = input_shape[3];
    maxpool_params->h_out    = output_shape[2];
    maxpool_params->w_out    = output_shape[3];

    *params = (void *) maxpool_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w)
{
    volatile maxpool2d_fp16_spatz_params_t *maxpool_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;
    uint32_t in_hw_len;
    uint32_t out_hw_len;
    uint32_t local_idx;

    maxpool_params = (volatile maxpool2d_fp16_spatz_params_t *) params;
    shard_X = maxpool_params->shard_X;
    shard_Y = maxpool_params->shard_Y;

    in_hw_len  = maxpool_params->h_in * maxpool_params->w_in;
    out_hw_len = maxpool_params->h_out * maxpool_params->w_out;

    local_idx = 0;
    for (uint32_t c = maxpool_params->c_start; c < (maxpool_params->c_start + maxpool_params->c_len); c++) {
        uint32_t c_base = c * in_hw_len;

        for (uint32_t i = 0; i < in_hw_len; i++) {
            uint32_t global_idx = c_base + i;
            uint32_t offset = local_idx * sizeof(float16);

            mmio_fp16(shard_X + offset) = X[global_idx];
            local_idx++;
        }
    }

    uint32_t total_out_elements = maxpool_params->c_len * out_hw_len;
    for (uint32_t i = 0; i < total_out_elements; i++) {
        uint32_t offset = i * sizeof(float16);
        mmio_fp16(shard_Y + offset) = 0;
    }

    maxpool_params->kernel_h  = kernel_h;
    maxpool_params->kernel_w  = kernel_w;
    maxpool_params->stride_h  = stride_h;
    maxpool_params->stride_w  = stride_w;
    maxpool_params->pad_h     = pad_h;
    maxpool_params->pad_w     = pad_w;

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

    spatz_run_task_with_params(MAXPOOL2D_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] Wait on Spatz task completion failed with error: %d\n", HID, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *Y)
{
    volatile maxpool2d_fp16_spatz_params_t *maxpool_params;
    uintptr_t shard_Y_base;
    uint32_t out_hw_len;
    uint32_t local_idx;

    maxpool_params = (volatile maxpool2d_fp16_spatz_params_t *) params;
    shard_Y_base = maxpool_params->shard_Y;

    out_hw_len = maxpool_params->h_out * maxpool_params->w_out;
    local_idx = 0;

    for (uint32_t c = 0; c < maxpool_params->c_len; c++) {
        uint32_t c_idx = maxpool_params->c_start + c;
        uint32_t c_base = c_idx * out_hw_len;

        for (uint32_t i = 0; i < out_hw_len; i++) {
            uint32_t global_idx = c_base + i;
            uint32_t offset = local_idx * sizeof(float16);

            Y[global_idx] = mmio_fp16(shard_Y_base + offset);
            local_idx++;
        }
    }

    return 0;
}

void MAGIA_maxpool2d_fp16_spatz(const float16* X, float16 *Y, uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t input_shape[4], uint32_t output_shape[4])
{
    int ret;
    volatile maxpool2d_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, input_shape, output_shape);
    if (ret != 0) {
        printf("[CV32 (%d)] L1 allocation failed with error: %d\n", HID, ret);
        return;
    }

    ret = init_input_params(params, X, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
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
