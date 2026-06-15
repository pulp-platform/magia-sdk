#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "concat_fp16_spatz.h"
#include "concat_fp16_spatz_params.h"
#include "concat_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int alloc_l1(void **params, uint32_t in0_transfer_len,  uint32_t in1_transfer_len, uint32_t iterations)
{
    volatile concat_fp16_spatz_params_t *concat_params;
    uintptr_t shard_input0;
    uintptr_t shard_input1;
    uintptr_t shard_output;

    size_t iter_start;
    size_t iter_end;
    size_t iter_len;
    size_t elems;
    size_t left;

    elems = iterations / NUM_HARTS;
    left = iterations % NUM_HARTS;

    iter_start = HID * elems + (HID < left ? HID : left);
    iter_end = iter_start + elems + (HID < left ? 1 : 0);
    iter_len = iter_end - iter_start;

    l1_alloc_init();

    concat_params = l1_alloc(sizeof(concat_fp16_spatz_params_t));
    if (!concat_params)
        return ENOMEM;

    shard_input0 = l1_alloc(iter_len * in0_transfer_len * sizeof(float16));
    if (!shard_input0)
        return ENOMEM;

    shard_input1 = l1_alloc(iter_len * in1_transfer_len * sizeof(float16));
    if (!shard_input1)
        return ENOMEM;

    shard_output = l1_alloc(iter_len * (in0_transfer_len + in1_transfer_len) * sizeof(float16));
    if (!shard_output)
        return ENOMEM;

    concat_params->shard_input0 = shard_input0;
    concat_params->shard_input1 = shard_input1;
    concat_params->shard_output = shard_output;
    concat_params->len_input0 = in0_transfer_len;
    concat_params->len_input1 = in1_transfer_len;
    concat_params->iter_start = iter_start;
    concat_params->iter_len = iter_len;

    *params = (void *) concat_params;

    return 0;
}

static int init_input_params(void *params, const float16 *input0, const float16 *input1)
{
    volatile concat_fp16_spatz_params_t *concat_params;
    uintptr_t shard_input0;
    uintptr_t shard_input1;
    uintptr_t shard_output;
    size_t iter_start;
    size_t iter_len;
    size_t iter_end;
    size_t len_in0;
    size_t len_in1;

    concat_params = (volatile concat_fp16_spatz_params_t *) params;

    shard_input0 = concat_params->shard_input0;
    shard_input1 = concat_params->shard_input1;
    shard_output = concat_params->shard_output;

    iter_start = concat_params->iter_start;
    iter_len = concat_params->iter_len;
    iter_end = iter_start + iter_len;
    len_in0 = concat_params->len_input0;
    len_in1 = concat_params->len_input1;

    uint32_t l1_in0_idx = 0;
    uint32_t l1_in1_idx = 0;
    uint32_t l1_out_idx = 0;

    for (uint32_t iter = iter_start; iter < iter_end; iter++) {
        uint32_t global_idx_in0 = iter * len_in0;
        uint32_t global_idx_in1 = iter * len_in1;
        uint32_t offset;

        for (uint32_t i = 0; i < len_in0; i++) {
            offset = l1_in0_idx * sizeof(float16);
            mmio_fp16(shard_input0 + offset) = input0[global_idx_in0 + i];
            l1_in0_idx++;
        }

        for (uint32_t i = 0; i < len_in1; i++) {
            offset = l1_in1_idx * sizeof(float16);
            mmio_fp16(shard_input1 + offset) = input1[global_idx_in1 + i];
            l1_in1_idx++;
        }

        uint32_t output_len = len_in0 + len_in1;
        for (uint32_t i = 0; i < output_len; i++) {
            offset = l1_out_idx * sizeof(float16);
            mmio_fp16(shard_output + offset) = 0;
            l1_out_idx++;
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

    spatz_init(SPATZ_BINARY_START);
    spatz_run_task_with_params(CONCAT_FP16_SPATZ_TASK, params);

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

    for (uint32_t iter = 0; iter < len; iter++) {
        uint32_t global_iter = start + iter;
        uint32_t global_idx_base = global_iter * out_len;

        for (uint32_t i = 0; i < out_len; i++) {
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

    ret = alloc_l1(&params, in0_transfer_len, in1_transfer_len, iterations);
    if (ret != 0) {
        printf("[CV32 (%d)] L1 allocation failed with error: %d\n", HID, ret);
        return;
    }

    ret = init_input_params(params, input0, input1);
    if (ret != 0) {
        printf("[CV32 (%d)] Params initialization failed with error: %d\n", HID, ret);
        return;
    }

    ret = offload_spatz_task(params);
    if (ret != 0) {
        printf("[CV32 (%d)] Spatz task offloading failed with error: %d\n", HID, ret);
        return;
    }

    ret = store_result(params, concat_result, iterations);
    if (ret != 0) {
        printf("[CV32 (%d)] Result write back failed with error: %d\n", HID, ret);
    }
}
