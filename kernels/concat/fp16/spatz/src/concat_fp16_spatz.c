#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "concat_fp16_spatz.h"
#include "concat_fp16_spatz_mem_layout.h"
#include "concat_fp16_spatz_params.h"
#include "concat_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int init_input_params(void *params, const float16 *input0, const float16 *input1, const uint32_t in0_transfer_len, const uint32_t in1_transfer_len, const uint32_t iterations)
{
    volatile concat_fp16_spatz_params_t *concat_params;
    uint32_t iter_start;
    uint32_t iter_end;
    uint32_t iter_len;
    uint32_t shard;
    uint32_t left;


    concat_params = (volatile concat_fp16_spatz_params_t *) params;

    shard = iterations / NUM_HARTS;
    left = iterations % NUM_HARTS;

    iter_start = HID * shard + (HID < left ? HID : left);
    iter_end = iter_start + shard + (HID < left ? 1 : 0);
    iter_len = iter_end - iter_start;

    uint32_t l1_in0_idx = 0;
    uint32_t l1_in1_idx = 0;
    uint32_t l1_out_idx = 0;

    for (int iter = iter_start; iter < iter_end; iter++) {
        uint32_t global_idx_in0 = iter * in0_transfer_len;
        uint32_t global_idx_in1 = iter * in1_transfer_len;
        uint32_t offset;

        for (int i = 0; i < in0_transfer_len; i++) {
            offset = l1_in0_idx * sizeof(float16);
            mmio_fp16(SHARD_INPUT0_BASE + offset) = input0[global_idx_in0 + i];
            l1_in0_idx++;
        }

        for (int i = 0; i < in1_transfer_len; i++) {
            offset = l1_in1_idx * sizeof(float16);
            mmio_fp16(SHARD_INPUT1_BASE + offset) = input1[global_idx_in1 + i];
            l1_in1_idx++;
        }

        uint32_t output_len = in0_transfer_len + in1_transfer_len;
        for (int i = 0; i < output_len; i++) {
            offset = l1_out_idx * sizeof(float16);
            mmio_fp16(SHARD_OUTPUT_BASE + offset) = 0;
            l1_out_idx++;
        }
    }

    concat_params->shard_input0 = SHARD_INPUT0_BASE;
    concat_params->shard_input1 = SHARD_INPUT1_BASE;
    concat_params->shard_output = SHARD_OUTPUT_BASE;
    concat_params->len_input0 = in0_transfer_len;
    concat_params->len_input1 = in1_transfer_len;
    concat_params->iter_start = iter_start;
    concat_params->iter_len = iter_len;

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
    spatz_run_task_with_params(CONCAT_FP16_SPATZ_TASK, CONCAT_FP16_SPATZ_PARAMS_BASE);

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

static int store_result(void *params, float16 *concat_result, const uint32_t iterations)
{
    volatile concat_fp16_spatz_params_t *concat_params;
    uint32_t shard_output_base;
    uint32_t in0_len;
    uint32_t in1_len;
    uint32_t out_len;
    uint32_t start;
    uint32_t len;

    concat_params = (volatile concat_fp16_spatz_params_t *) params;
    shard_output_base = concat_params->shard_output;
    in0_len = concat_params->len_input0;
    in1_len = concat_params->len_input1;
    start = concat_params->iter_start;
    len = concat_params->iter_len;
    out_len = in0_len + in1_len;

    uint32_t out_idx = 0;

    for (int iter = 0; iter < len; iter++) {
        uint32_t global_iter = start + iter;
        uint32_t global_idx_base = global_iter * out_len;

        for (int i = 0; i < out_len; i++) {
            uint32_t offset = out_idx * sizeof(float16);
            concat_result[global_idx_base + i] = mmio_fp16(shard_output_base + offset);
            out_idx++;
        }
    }

    return 0;
}

void MAGIA_concat_fp16_spatz(const float16 *input0, const float16 *input1, float16 *concat_result, uint32_t in0_transfer_len, uint32_t in1_transfer_len, uint32_t axis, uint32_t iterations)
{
    int ret;
    volatile concat_fp16_spatz_params_t *params;

    params = (volatile concat_fp16_spatz_params_t *) CONCAT_FP16_SPATZ_PARAMS_BASE;

    ret = init_input_params(params, input0, input1, in0_transfer_len, in1_transfer_len, iterations);
    if (ret != 0) {
        printf("[CV32 (%d)] Params initialization failed with error: %d\n", HID, ret);
        return;
    }

    ret = offload_spatz_task();
    if (ret != 0) {
        printf("[CV32 (%d)] Spatz task offloading failed with error: %d\n", HID, ret);
        return;
    }

    ret = store_result(params, concat_result, iterations);
    if (ret != 0) {
        printf("[CV32 (%d)] Result write back failed with error: %d\n", HID, ret);
    }
}
