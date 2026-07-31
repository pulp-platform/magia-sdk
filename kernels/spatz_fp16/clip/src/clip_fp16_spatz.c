#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "clip_fp16_spatz.h"
#include "clip_fp16_spatz_params.h"
#include "clip_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "clip_fp16_spatz"

static int alloc_l1(void **params, uint32_t size)
{
    volatile clip_fp16_spatz_params_t *clip_params;
    uintptr_t shard_input;
    uintptr_t shard_output;
    uintptr_t min;
    uintptr_t max;

    size_t shard_start;
    size_t shard_end;
    size_t shard_len;
    size_t elems;
    size_t left;

    elems = size / NUM_HARTS;
    left = size % NUM_HARTS;
    shard_start = HID * elems + (HID < left ? HID : left);
    shard_end = shard_start + elems + (HID < left ? 1 : 0);
    shard_len = shard_end - shard_start;

    l1_alloc_init();

    clip_params = l1_alloc(sizeof(clip_fp16_spatz_params_t));
    if (!clip_params)
        return ENOMEM;

    shard_input = l1_alloc(shard_len * sizeof(float16));
    if (!shard_input)
        return ENOMEM;

    shard_output = l1_alloc(shard_len * sizeof(float16));
    if (!shard_output)
        return ENOMEM;

    min = l1_alloc(sizeof(float16));
    if (!min)
        return ENOMEM;

    max = l1_alloc(sizeof(float16));
    if (!max)
        return ENOMEM;

    clip_params->shard_input = shard_input;
    clip_params->shard_output = shard_output;
    clip_params->start = shard_start;
    clip_params->len = shard_len;
    clip_params->end = shard_end;
    clip_params->min = min;
    clip_params->max = max;

    *params = (void *) clip_params;

    return 0;
}

static int init_input_params(void *params, const float16 *input, float16 minimum, float16 maximum)
{
    volatile clip_fp16_spatz_params_t *clip_params;
    uintptr_t shard_input;
    uintptr_t shard_output;
    uintptr_t min;
    uintptr_t max;
    uint32_t start;
    uint32_t len;

    clip_params = (volatile clip_fp16_spatz_params_t *) params;

    shard_input = clip_params->shard_input;
    shard_output = clip_params->shard_output;
    min = clip_params->min;
    max = clip_params->max;

    start = clip_params->start;
    len = clip_params->len;

    for (uint32_t i = 0; i < len; i++) {
        uint32_t global_idx = start + i;
        uint32_t offset = i * sizeof(float16);

        mmio_fp16(shard_input + offset) = input[global_idx];
        mmio_fp16(shard_output + offset) = 0;
    }

    mmio_fp16(min) = minimum;
    mmio_fp16(max) = maximum;

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

    spatz_run_task_with_params(CLIP_FP16_SPATZ_TASK, params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void* params, float16 *dst)
{
    volatile clip_fp16_spatz_params_t *clip_params;
    uint32_t shard_out_base;
    uint32_t start;
    uint32_t len;

    clip_params = (volatile clip_fp16_spatz_params_t *) params;
    shard_out_base = clip_params->shard_output;
    start = clip_params->start;
    len = clip_params->len;

    for (uint32_t i = 0; i < len; i++) {
        uint32_t global_idx = start + i;
        uint32_t offset = i * sizeof(float16);
        dst[global_idx] = mmio_fp16(shard_out_base + offset);
    }

    return 0;
}

void MAGIA_clip_fp16_spatz(const float16 *input, float16 *output, float16 min, float16 max, uint32_t size)
{
    int ret;
    volatile clip_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, size);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params(params, input, min, max);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task(params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result(params, output);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
