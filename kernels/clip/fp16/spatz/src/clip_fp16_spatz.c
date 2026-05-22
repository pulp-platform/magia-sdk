#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "clip_fp16_spatz.h"
#include "clip_fp16_spatz_mem_layout.h"
#include "clip_fp16_spatz_params.h"
#include "clip_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int init_input_params(void *params, const float16 *input, float16 min, float16 max, uint32_t size)
{
    volatile clip_fp16_spatz_params_t *clip_params;
    uint32_t shard;
    uint32_t start;
    uint32_t left;
    uint32_t end;
    uint32_t len;

    clip_params = (volatile clip_fp16_spatz_params_t *) params;

    shard = size / NUM_HARTS;
    left = size % NUM_HARTS;
    start = HID * shard + (HID < left ? HID : left);
    end   = start + shard + (HID < left ? 1 : 0);
    len   = end - start;

    for (int i = 0; i < len; i++) {
        uint32_t global_idx = start + i;
        uint32_t offset = i * sizeof(float16);

        mmio_fp16(SHARD_IN_BASE + offset) = input[global_idx];
        mmio_fp16(SHARD_OUT_BASE + offset) = 0;
    }

    mmio_fp16(MIN_BASE) = min;
    mmio_fp16(MAX_BASE) = max;

    clip_params->shard_input = SHARD_IN_BASE;
    clip_params->shard_output = SHARD_OUT_BASE;
    clip_params->min = MIN_BASE;
    clip_params->max = MAX_BASE;
    clip_params->start = start;
    clip_params->len = len;
    clip_params->end = end;

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
    spatz_run_task_with_params(CLIP_FP16_SPATZ_TASK, CLIP_FP16_SPATZ_PARAMS_BASE);

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
    volatile clip_fp16_spatz_params_t *clip_params;
    uint32_t shard_out_base;
    uint32_t start;
    uint32_t len;

    clip_params = (volatile clip_fp16_spatz_params_t *) params;
    shard_out_base = clip_params->shard_output;
    start = clip_params->start;
    len = clip_params->len;

    for (int i = 0; i < len; i++) {
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

    params = (volatile clip_fp16_spatz_params_t *) CLIP_FP16_SPATZ_PARAMS_BASE;

    ret = init_input_params(params, input, min, max, size);
    if (ret != 0) {
        printf("[CV32 (%d)] Params initialization failed with error: %d\n", HID, ret);
        return;
    }

    ret = offload_spatz_task();
    if (ret != 0) {
        printf("[CV32 (%d)] Spatz task offloading failed with error: %d\n", HID, ret);
        return;
    }

    ret = store_result(params, output);
    if (ret != 0) {
        printf("[CV32 (%d)] Result write back failed with error: %d\n", HID, ret);
    }
}
