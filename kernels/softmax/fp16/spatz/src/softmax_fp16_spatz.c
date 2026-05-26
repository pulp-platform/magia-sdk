#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "softmax_fp16_spatz.h"
#include "softmax_fp16_spatz_mem_layout.h"
#include "softmax_fp16_spatz_params.h"
#include "softmax_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int init_input_params(void *params, const float16 *input)
{
    volatile softmax_fp16_spatz_params_t *softmax_params;
    uint32_t shard;
    uint32_t left;
    uint32_t r_start;
    uint32_t r_end;
    uint32_t r_len;
    uint32_t w_len;
    uint32_t local_idx;

    softmax_params = (volatile softmax_fp16_spatz_params_t *) params;

    shard = TOTAL_ROWS / NUM_HARTS;
    left  = TOTAL_ROWS % NUM_HARTS;

    r_start = HID * shard + (HID < left ? HID : left);
    r_end   = r_start + shard + (HID < left ? 1 : 0);
    r_len   = r_end - r_start;
    w_len   = ROW_LEN;

    local_idx = 0;
    for (int r = r_start; r < r_end; r++) {
        uint32_t r_base;

        r_base = r * w_len;

        for (int i = 0; i < w_len; i++) {
            uint32_t global_idx;
            uint32_t offset;

            global_idx = r_base + i;
            offset = local_idx * sizeof(float16);

            mmio_fp16(SHARD_INPUT_BASE + offset) = input[global_idx];
            mmio_fp16(SHARD_OUTPUT_BASE + offset) = 0;

            local_idx++;
        }
    }

    softmax_params->shard_input = SHARD_INPUT_BASE;
    softmax_params->shard_output = SHARD_OUTPUT_BASE;
    softmax_params->r_start = r_start;
    softmax_params->r_len   = r_len;
    softmax_params->w_len   = w_len;

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
    spatz_run_task_with_params(SOFTMAX_FP16_SPATZ_TASK, SOFTMAX_FP16_SPATZ_PARAMS_BASE);

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

static int store_result(void *params, float16 *output)
{
    volatile softmax_fp16_spatz_params_t *softmax_params;
    uint32_t shard_out_base;
    uint32_t local_idx;

    softmax_params = (volatile softmax_fp16_spatz_params_t *) params;
    shard_out_base = softmax_params->shard_output;
    local_idx = 0;

    for (int r = 0; r < softmax_params->r_len; r++) {
        uint32_t r_idx;
        uint32_t r_base;

        r_idx = softmax_params->r_start + r;
        r_base = r_idx * softmax_params->w_len;

        for (int i = 0; i < softmax_params->w_len; i++) {
            uint32_t global_idx;
            uint32_t offset;

            global_idx = r_base + i;
            offset = local_idx * sizeof(float16);

            output[global_idx] = mmio_fp16(shard_out_base + offset);

            local_idx++;
        }
    }

    return 0;
}

void MAGIA_softmax_fp16_spatz(const float16 *input, float16 *output)
{
    int ret;
    volatile softmax_fp16_spatz_params_t *params;

    params = (volatile softmax_fp16_spatz_params_t *) SOFTMAX_FP16_SPATZ_PARAMS_BASE;

    ret = init_input_params(params, input);
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
