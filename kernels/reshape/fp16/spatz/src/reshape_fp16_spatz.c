#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "reshape_fp16_spatz.h"
#include "reshape_fp16_spatz_params.h"
#include "reshape_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "reshape_fp16_spatz"

static int alloc_l1(void **params, uint32_t total_elements)
{
    volatile reshape_fp16_spatz_params_t *reshape_params;
    uint32_t shard_X;
    uint32_t shard_Y;

    size_t start_idx;
    size_t end_idx;
    size_t len;
    size_t elems;
    size_t left;

    elems = total_elements / NUM_HARTS;
    left = total_elements % NUM_HARTS;

    start_idx = HID * elems + (HID < left ? HID : left);
    end_idx = start_idx + elems + (HID < left ? 1 : 0);
    len = end_idx - start_idx;

    l1_alloc_init();

    reshape_params = l1_alloc(sizeof(reshape_fp16_spatz_params_t));
    if (!reshape_params)
        return ENOMEM;

    shard_X = l1_alloc(len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(len * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    reshape_params->shard_X = shard_X;
    reshape_params->shard_Y = shard_Y;
    reshape_params->start_idx = start_idx;
    reshape_params->len = len;

    *params = (void *) reshape_params;

    return 0;
}

static int init_input_params(void *params, const float16 *data)
{
    volatile reshape_fp16_spatz_params_t *reshape_params;
    uint32_t shard_X;
    uint32_t shard_Y;

    size_t start_idx;
    size_t len;

    reshape_params = (volatile reshape_fp16_spatz_params_t *) params;

    shard_X = reshape_params->shard_X;
    shard_Y = reshape_params->shard_Y;
    start_idx = reshape_params->start_idx;
    len = reshape_params->len;

    for (uint32_t i = 0; i < len; i++) {
        uint32_t global_idx;
        uint32_t offset;

        global_idx = start_idx + i;
        offset = i * sizeof(float16);

        mmio_fp16(shard_X + offset) = data[global_idx];
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

    spatz_run_task_with_params(RESHAPE_FP16_SPATZ_TASK, params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *reshaped)
{
    volatile reshape_fp16_spatz_params_t *reshape_params;
    uint32_t shard_Y_base;
    uint32_t start_idx;
    uint32_t len;

    reshape_params = (volatile reshape_fp16_spatz_params_t *) params;
    shard_Y_base = reshape_params->shard_Y;
    start_idx = reshape_params->start_idx;
    len = reshape_params->len;

    for (uint32_t i = 0; i < len; i++) {
        uint32_t global_idx;
        uint32_t offset;

        global_idx = start_idx + i;
        offset = i * sizeof(float16);

        reshaped[global_idx] = mmio_fp16(shard_Y_base + offset);
    }

    return 0;
}

void MAGIA_reshape_fp16_spatz(const float16 *data, float16 *reshaped, uint32_t total_elements)
{
    int ret;
    volatile reshape_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, total_elements);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params(params, data);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task(params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result(params, reshaped);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
