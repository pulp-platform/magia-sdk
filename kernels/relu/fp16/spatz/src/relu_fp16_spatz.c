#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "relu_fp16_spatz.h"
#include "relu_fp16_spatz_params.h"
#include "relu_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int alloc_l1(void **params, uint32_t size)
{
    volatile relu_fp16_spatz_params_t *relu_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;

    uint32_t shard;
    uint32_t left;
    uint32_t start;
    uint32_t end;
    uint32_t len;

    shard = size / NUM_HARTS;
    left  = size % NUM_HARTS;
    start = HID * shard + (HID < left ? HID : left);
    end   = start + shard + (HID < left ? 1 : 0);
    len   = end - start;

    l1_alloc_init();

    relu_params = l1_alloc(sizeof(relu_fp16_spatz_params_t));
    if (!relu_params)
        return ENOMEM;

    shard_X = l1_alloc(len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(len * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    relu_params->shard_X = shard_X;
    relu_params->shard_Y = shard_Y;
    relu_params->start   = start;
    relu_params->len     = len;
    relu_params->end     = end;

    *params = (void *) relu_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile relu_fp16_spatz_params_t *relu_params;
    uintptr_t shard_X_base;
    uintptr_t shard_Y_base;
    uint32_t start;
    uint32_t len;

    relu_params = (volatile relu_fp16_spatz_params_t *) params;
    shard_X_base = relu_params->shard_X;
    shard_Y_base = relu_params->shard_Y;

    start = relu_params->start;
    len   = relu_params->len;

    for (uint32_t i = 0; i < len; i++) {
        uint32_t global_idx = start + i;
        uint32_t offset = i * sizeof(float16);

        mmio_fp16(shard_X_base + offset) = X[global_idx];
        mmio_fp16(shard_Y_base + offset) = 0;
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
    spatz_run_task_with_params(RELU_FP16_SPATZ_TASK, (uint32_t)params);

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

static int store_result(void *params, float16 *dst)
{
    volatile relu_fp16_spatz_params_t *relu_params;
    uintptr_t shard_Y_base;
    uint32_t start;
    uint32_t len;

    relu_params = (volatile relu_fp16_spatz_params_t *) params;
    shard_Y_base = relu_params->shard_Y;
    start = relu_params->start;
    len = relu_params->len;

    for (uint32_t i = 0; i < len; i++) {
        uint32_t global_idx = start + i;
        uint32_t offset = i * sizeof(float16);
        dst[global_idx] = mmio_fp16(shard_Y_base + offset);
    }

    return 0;
}

void MAGIA_relu_fp16_spatz(const float16 *X, float16 *Y, uint32_t size)
{
    int ret;
    volatile relu_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, size);
    if (ret != 0) {
        printf("[CV32 (%d)] L1 allocation failed with error: %d\n", HID, ret);
        return;
    }

    ret = init_input_params((void *)params, X);
    if (ret != 0) {
        printf("[CV32 (%d)] Params initialization failed with error: %d\n", HID, ret);
        return;
    }

    ret = offload_spatz_task((void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] Spatz task offloading failed with error: %d\n", HID, ret);
        return;
    }

    ret = store_result((void *)params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] Result write back failed with error: %d\n", HID, ret);
    }
}
