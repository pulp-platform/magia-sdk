#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "sigmoid_fp16_spatz.h"
#include "sigmoid_fp16_spatz_mem_layout.h"
#include "sigmoid_fp16_spatz_params.h"
#include "sigmoid_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int init_input_params(void *params, const float16 *X, uint32_t size)
{
    volatile sigmoid_fp16_spatz_params_t *sigmoid_params;
    uint32_t shard;
    uint32_t start;
    uint32_t left;
    uint32_t end;
    uint32_t len;
    uint32_t hid;

    sigmoid_params = (volatile sigmoid_fp16_spatz_params_t *) params;

    shard = size / NUM_HARTS;
    left  = size % NUM_HARTS;
    start = HID * shard + (HID < left ? HID : left);
    end   = start + shard + (HID < left ? 1 : 0);
    len   = end - start;

    for (int i = 0; i < len; i++) {
        uint32_t global_idx = start + i;
        uint32_t offset = i * sizeof(float16);

        mmio_fp16(SHARD_X_BASE + offset) = X[global_idx];
        mmio_fp16(SHARD_Y_BASE + offset) = 0;
    }

    sigmoid_params->shard_X = SHARD_X_BASE;
    sigmoid_params->shard_Y = SHARD_Y_BASE;
    sigmoid_params->start   = start;
    sigmoid_params->len     = len;
    sigmoid_params->end     = end;

    return 0;
}

static int offload_spatz_task()
{
    eu_controller_t eu_ctrl;
    eu_config_t eu_cfg;
    int ret;

    eu_cfg.hartid = HID;
    eu_ctrl.base = NULL;
    eu_ctrl.cfg = &eu_cfg;
    eu_ctrl.api = &eu_api;

    spatz_init(SPATZ_BINARY_START);
    spatz_run_task_with_params(SIGMOID_FP16_SPATZ_TASK, SIGMOID_FP16_SPATZ_PARAMS_BASE);

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
    volatile sigmoid_fp16_spatz_params_t *sigmoid_params;
    uint32_t shard_Y_base;
    uint32_t start;
    uint32_t len;

    sigmoid_params = (volatile sigmoid_fp16_spatz_params_t *) params;
    shard_Y_base = sigmoid_params->shard_Y;
    start = sigmoid_params->start;
    len = sigmoid_params->len;

    for (int i = 0; i < len; i++) {
        uint32_t global_idx = start + i;
        uint32_t offset = i * sizeof(float16);
        dst[global_idx] = mmio_fp16(shard_Y_base + offset);
    }

    return 0;
}

void MAGIA_sigmoid_fp16_spatz(const float16 *X, float16 *Y, uint32_t size)
{
    int ret;
    volatile sigmoid_fp16_spatz_params_t *params;

    params = (volatile sigmoid_fp16_spatz_params_t *) SIGMOID_FP16_SPATZ_PARAMS_BASE;

    ret = init_input_params(params, X, size);
    if (ret != 0) {
        printf("[CV32 (%d) Params initialization failed with error: %d\n]", HID, ret);
        return;
    }

    ret = offload_spatz_task();
    if (ret != 0) {
        printf("[CV32 (%d)] Spatz task offloading failed with error: %d\n", HID, ret);
        return;
    }

    ret = store_result(params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] Result write back failed with error: %d\n", HID, ret);
    }
}
