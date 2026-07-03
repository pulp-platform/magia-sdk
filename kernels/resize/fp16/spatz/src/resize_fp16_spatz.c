#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "resize_fp16_spatz.h"
#include "resize_fp16_spatz_params.h"
#include "resize_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int alloc_l1(void **params, uint32_t batch_size, uint32_t channels, uint32_t in_h, uint32_t in_w, uint32_t out_h, uint32_t out_w)
{
    volatile resize_fp16_spatz_params_t *resize_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;

    size_t it_start;
    size_t it_end;
    size_t it_len;
    size_t elems;
    size_t left;
    size_t total_iterations;

    total_iterations = batch_size * channels;
    elems = total_iterations / NUM_HARTS;
    left = total_iterations % NUM_HARTS;

    it_start = HID * elems + (HID < left ? HID : left);
    it_end = it_start + elems + (HID < left ? 1 : 0);
    it_len = it_end - it_start;

    l1_alloc_init();

    resize_params = l1_alloc(sizeof(resize_fp16_spatz_params_t));
    if (!resize_params)
        return ENOMEM;

    shard_X = l1_alloc(it_len * in_h * in_w * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(it_len * out_h * out_w * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    resize_params->shard_X = shard_X;
    resize_params->shard_Y = shard_Y;
    resize_params->in_h = in_h;
    resize_params->in_w = in_w;
    resize_params->out_h = out_h;
    resize_params->out_w = out_w;
    resize_params->iteration_start = it_start;
    resize_params->iteration_len = it_len;

    *params = (void *) resize_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile resize_fp16_spatz_params_t *resize_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;
    size_t it_start;
    size_t it_len;
    size_t in_hw;
    size_t out_hw;

    resize_params = (volatile resize_fp16_spatz_params_t *) params;

    shard_X  = resize_params->shard_X;
    shard_Y  = resize_params->shard_Y;
    it_start = resize_params->iteration_start;
    it_len   = resize_params->iteration_len;

    in_hw  = resize_params->in_h * resize_params->in_w;
    out_hw = resize_params->out_h * resize_params->out_w;

    uint32_t local_idx = 0;
    for (uint32_t i = 0; i < it_len; i++) {
        uint32_t global_channel_idx = it_start + i;
        uint32_t global_base = global_channel_idx * in_hw;

        for (uint32_t pixel = 0; pixel < in_hw; pixel++) {
            uint32_t global_idx = global_base + pixel;
            uint32_t offset = local_idx * sizeof(float16);

            mmio_fp16(shard_X + offset) = X[global_idx];
            local_idx++;
        }
    }

    size_t total_out_elements = it_len * out_hw;
    for (uint32_t i = 0; i < total_out_elements; i++) {
        uint32_t offset = i * sizeof(float16);
        mmio_fp16(shard_Y + offset) = 0;
    }

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
    spatz_run_task_with_params(RESIZE_FP16_SPATZ_TASK, params);

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
    volatile resize_fp16_spatz_params_t *resize_params;
    uintptr_t shard_Y_base;
    size_t it_start;
    size_t it_len;
    size_t out_hw;

    resize_params = (volatile resize_fp16_spatz_params_t *) params;
    shard_Y_base  = resize_params->shard_Y;
    it_start      = resize_params->iteration_start;
    it_len        = resize_params->iteration_len;
    out_hw        = resize_params->out_h * resize_params->out_w;

    uint32_t local_idx = 0;
    for (uint32_t i = 0; i < it_len; i++) {
        uint32_t global_channel_idx = it_start + i;
        uint32_t global_base = global_channel_idx * out_hw;

        for (uint32_t pixel = 0; pixel < out_hw; pixel++) {
            uint32_t global_idx = global_base + pixel;
            uint32_t offset = local_idx * sizeof(float16);

            Y[global_idx] = mmio_fp16(shard_Y_base + offset);
            local_idx++;
        }
    }

    return 0;
}

void MAGIA_resize_fp16_spatz(const float16 *X, float16 *Y, uint32_t batch_size, uint32_t channels, uint32_t in_h, uint32_t in_w, uint32_t out_h, uint32_t out_w)
{
    int ret;
    volatile resize_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, batch_size, channels, in_h, in_w, out_h, out_w);
    if (ret != 0) {
        printf("[CV32 (%d)] L1 allocation failed with error: %d\n", HID, ret);
        return;
    }

    ret = init_input_params(params, X);
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
