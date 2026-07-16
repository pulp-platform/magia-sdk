#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "ceil_fp16_spatz.h"
#include "ceil_fp16_spatz_params.h"
#include "ceil_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int alloc_l1(void **params, size_t size)
{
    volatile ceil_fp16_spatz_params_t *ceil_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;

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

    ceil_params = l1_alloc(sizeof(ceil_fp16_spatz_params_t));
    if (!ceil_params)
        return ENOMEM;

    shard_X = l1_alloc(shard_len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(shard_len * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    ceil_params->shard_X = shard_X;
    ceil_params->shard_Y = shard_Y;
    ceil_params->start = shard_start;
    ceil_params->len = shard_len;
    ceil_params->end = shard_end;

    *params = (void *) ceil_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile ceil_fp16_spatz_params_t *ceil_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;
    uint32_t start;
    uint32_t len;

    ceil_params = (volatile ceil_fp16_spatz_params_t *) params;

    shard_X = ceil_params->shard_X;
    shard_Y = ceil_params->shard_Y;

    start = ceil_params->start;
    len = ceil_params->len;

    for (int i = 0; i < len; i++) {
        uint32_t global_idx = start + i;
        uint32_t offset = i * sizeof(float16);

        mmio_fp16(shard_X + offset) = X[global_idx];
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

    spatz_run_task_with_params(CEIL_FP16_SPATZ_TASK, params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] Wait on Spatz task completion failed with error: %d\n", HID, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void* params, float16 *dst)
{
    volatile ceil_fp16_spatz_params_t *ceil_params;
    uint32_t shard_Y_base;
    uint32_t start;
    uint32_t len;

    ceil_params = (volatile ceil_fp16_spatz_params_t *) params;
    shard_Y_base = ceil_params->shard_Y;
    start = ceil_params->start;
    len = ceil_params->len;

    for (int i = 0; i < len; i++) {
        uint32_t global_idx = start + i;
        uint32_t offset = i * sizeof(float16);
        dst[global_idx] = mmio_fp16(shard_Y_base + offset);
    }

    return 0;
}


void MAGIA_ceil_fp16_spatz(const float16 *X, float16 *Y, uint32_t size)
{
    int ret;
    volatile ceil_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, size);
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
