#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "relu_fp16_spatz.h"
#include "relu_fp16_spatz_mem_layout.h"
#include "relu_fp16_spatz_params.h"
#include "relu_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int init_input_params(void *params, const float16 *X, uint32_t size)
{
    volatile relu_fp16_spatz_params_t *relu_params;
    uint32_t shard;
    uint32_t start;
    uint32_t left;
    uint32_t end;
    uint32_t len;
    uint32_t hid;

    relu_params = (volatile relu_fp16_spatz_params_t *) params;

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

    relu_params->shard_X = SHARD_X_BASE;
    relu_params->shard_Y = SHARD_Y_BASE;
    relu_params->start   = start;
    relu_params->len     = len;
    relu_params->end     = end;

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
    spatz_run_task_with_params(RELU_FP16_SPATZ_TASK, RELU_FP16_SPATZ_PARAMS_BASE);

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
    volatile relu_fp16_spatz_params_t *relu_params;
    uint32_t shard_Y_base;
    uint32_t start;
    uint32_t len;

    relu_params = (volatile relu_fp16_spatz_params_t *) params;
    shard_Y_base = relu_params->shard_Y;
    start = relu_params->start;
    len = relu_params->len;

    for (int i = 0; i < len; i++) {
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

    params = (volatile relu_fp16_spatz_params_t *) RELU_FP16_SPATZ_PARAMS_BASE;

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
