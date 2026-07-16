#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "softmax_fp16_spatz.h"
#include "softmax_fp16_spatz_params.h"
#include "softmax_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "softmax_fp16_spatz"

static int alloc_l1(void **params, uint32_t input_shape[4])
{
    volatile softmax_fp16_spatz_params_t *softmax_params;
    uintptr_t shard_input;
    uintptr_t shard_output;

    uint32_t total_rows;
    uint32_t row_len;
    uint32_t shard;
    uint32_t left;
    uint32_t r_start;
    uint32_t r_end;
    uint32_t r_len;

    total_rows = input_shape[0] * input_shape[1] * input_shape[2];
    row_len    = input_shape[3];

    shard = total_rows / NUM_HARTS;
    left  = total_rows % NUM_HARTS;

    r_start = HID * shard + (HID < left ? HID : left);
    r_end   = r_start + shard + (HID < left ? 1 : 0);
    r_len   = r_end - r_start;

    l1_alloc_init();

    softmax_params = l1_alloc(sizeof(softmax_fp16_spatz_params_t));
    if (!softmax_params)
        return ENOMEM;

    shard_input = l1_alloc(r_len * row_len * sizeof(float16));
    if (!shard_input)
        return ENOMEM;

    shard_output = l1_alloc(r_len * row_len * sizeof(float16));
    if (!shard_output)
        return ENOMEM;

    softmax_params->shard_input  = shard_input;
    softmax_params->shard_output = shard_output;
    softmax_params->r_start      = r_start;
    softmax_params->r_len        = r_len;
    softmax_params->w_len        = row_len;

    *params = (void *) softmax_params;

    return 0;
}

static int init_input_params(void *params, const float16 *input)
{
    volatile softmax_fp16_spatz_params_t *softmax_params;
    uintptr_t shard_input_base;
    uintptr_t shard_output_base;
    uint32_t r_end;
    uint32_t local_idx;

    softmax_params = (volatile softmax_fp16_spatz_params_t *) params;
    shard_input_base  = softmax_params->shard_input;
    shard_output_base = softmax_params->shard_output;
    r_end             = softmax_params->r_start + softmax_params->r_len;

    local_idx = 0;
    for (uint32_t r = softmax_params->r_start; r < r_end; r++) {
        uint32_t r_base = r * softmax_params->w_len;

        for (uint32_t i = 0; i < softmax_params->w_len; i++) {
            uint32_t global_idx = r_base + i;
            uint32_t offset = local_idx * sizeof(float16);

            mmio_fp16(shard_input_base + offset) = input[global_idx];
            mmio_fp16(shard_output_base + offset) = 0;

            local_idx++;
        }
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

    spatz_run_task_with_params(SOFTMAX_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *output)
{
    volatile softmax_fp16_spatz_params_t *softmax_params;
    uintptr_t shard_out_base;
    uint32_t local_idx;

    softmax_params = (volatile softmax_fp16_spatz_params_t *) params;
    shard_out_base = softmax_params->shard_output;
    local_idx = 0;

    for (uint32_t r = 0; r < softmax_params->r_len; r++) {
        uint32_t r_idx = softmax_params->r_start + r;
        uint32_t r_base = r_idx * softmax_params->w_len;

        for (uint32_t i = 0; i < softmax_params->w_len; i++) {
            uint32_t global_idx = r_base + i;
            uint32_t offset = local_idx * sizeof(float16);

            output[global_idx] = mmio_fp16(shard_out_base + offset);

            local_idx++;
        }
    }

    return 0;
}

void MAGIA_softmax_fp16_spatz(const float16 *input, float16 *output, uint32_t input_shape[4])
{
    int ret;
    volatile softmax_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, input_shape);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, input);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task((void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, output);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
