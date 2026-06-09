#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "selu_fp16_spatz.h"
#include "selu_fp16_spatz_params.h"
#include "selu_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int allocate_l1(void **params, uint32_t size)
{
    volatile selu_fp16_spatz_params_t *selu_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;
    uintptr_t alpha;
    uintptr_t gamma;

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

    selu_params = l1_alloc(sizeof(selu_fp16_spatz_params_t));
    if (!selu_params)
        return ENOMEM;

    shard_X = l1_alloc(shard_len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(shard_len * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    alpha = l1_alloc(sizeof(float16));
    if (!alpha)
        return ENOMEM;

    gamma = l1_alloc(sizeof(float16));
    if (!gamma)
        return ENOMEM;

    selu_params->shard_X = shard_X;
    selu_params->shard_Y = shard_Y;
    selu_params->alpha = alpha;
    selu_params->gamma = gamma;
    selu_params->start = shard_start;
    selu_params->end = shard_end;
    selu_params->len = shard_len;

    *params = (void *) selu_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, float16 a, float16 g)
{
    volatile selu_fp16_spatz_params_t *selu_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;
    uintptr_t alpha;
    uintptr_t gamma;
    size_t start;
    size_t len;

    selu_params = (volatile selu_fp16_spatz_params_t *) params;

    shard_X = selu_params->shard_X;
    shard_Y = selu_params->shard_Y;
    alpha = selu_params->alpha;
    gamma = selu_params->gamma;

    start = selu_params->start;
    len = selu_params->len;

    for (int i = 0; i < selu_params->len; i++) {
        uint32_t global_idx = start + i;
        uint32_t offset = i * sizeof(float16);

        mmio_fp16(shard_X + offset) = X[global_idx];
        mmio_fp16(shard_Y + offset) = 0;
    }

    mmio_fp16(alpha) = a;
    mmio_fp16(gamma) = g;

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
    spatz_run_task_with_params(SELU_FP16_SPATZ_TASK, params);

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

static int store_result(void* params, float16 *dst)
{
    volatile selu_fp16_spatz_params_t *selu_params;
    uint32_t shard_Y;
    uint32_t start;
    uint32_t len;

    selu_params = (volatile selu_fp16_spatz_params_t *) params;
    shard_Y = selu_params->shard_Y;
    start = selu_params->start;
    len = selu_params->len;

    for (int i = 0; i < len; i++) {
        uint32_t global_idx = start + i;
        uint32_t offset = i * sizeof(float16);
        dst[global_idx] = mmio_fp16(shard_Y + offset);
    }

    return 0;
}

void MAGIA_selu_fp16_spatz(const float16 *X, float16 *Y, uint32_t size, float16 alpha, float16 gamma)
{
    int ret;
    volatile selu_fp16_spatz_params_t *params;

    ret = allocate_l1(&params, size);
    if (ret != 0) {
        printf("[CV32 (%d)] L1 allocation failed with error: %d\n", HID, ret);
        return;
    }

    ret = init_input_params(params, X, alpha, gamma);
    if (ret != 0) {
        printf("[CV32 (%d) Params initialization failed with error: %d\n]", HID, ret);
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
