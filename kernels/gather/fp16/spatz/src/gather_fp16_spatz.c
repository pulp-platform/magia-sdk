#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "gather_fp16_spatz.h"
#include "gather_fp16_spatz_mem_layout.h"
#include "gather_fp16_spatz_params.h"
#include "gather_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int init_input_params(void *params, const float16 *data, uint32_t batch, uint32_t gather_dim_size, uint32_t axis_length, uint32_t index)
{
    volatile gather_fp16_spatz_params_t *gather_params;
    uint32_t batch_start;
    uint32_t batch_end;
    uint32_t batch_len;
    uint32_t shard;
    uint32_t left;

    gather_params = (volatile gather_fp16_spatz_params_t *) params;

    shard = batch / NUM_HARTS;
    left = batch % NUM_HARTS;

    batch_start = HID * shard + (HID < left ? HID : left);
    batch_end = batch_start + shard + (HID < left ? 1 : 0);
    batch_len = batch_end - batch_start;

    uint32_t l1_in_idx = 0;
    uint32_t l1_out_idx = 0;
    uint32_t in_batch_stride = gather_dim_size * axis_length;

    for (int b = batch_start; b < batch_end; b++) {
        uint32_t global_idx_in = b * in_batch_stride;
        uint32_t offset;

        for (int i = 0; i < in_batch_stride; i++) {
            offset = l1_in_idx * sizeof(float16);
            mmio_fp16(SHARD_INPUT_BASE + offset) = data[global_idx_in + i];
            l1_in_idx++;
        }

        for (int i = 0; i < axis_length; i++) {
            offset = l1_out_idx * sizeof(float16);
            mmio_fp16(SHARD_OUTPUT_BASE + offset) = 0;
            l1_out_idx++;
        }
    }

    gather_params->shard_input = SHARD_INPUT_BASE;
    gather_params->shard_output = SHARD_OUTPUT_BASE;
    gather_params->gather_dim_size = gather_dim_size;
    gather_params->axis_length = axis_length;
    gather_params->batch_start = batch_start;
    gather_params->batch_len = batch_len;
    gather_params->index = index;

    return 0;
}

static int offload_spatz_task(void)
{
    eu_controller_t eu_ctrl;
    eu_config_t eu_cfg;
    int ret;

    eu_cfg.hartid = HID;
    eu_ctrl.base = NULL;
    eu_ctrl.cfg = &eu_cfg;
    eu_ctrl.api = &eu_api;

    spatz_init(SPATZ_BINARY_START);
    spatz_run_task_with_params(GATHER_FP16_SPATZ_TASK, GATHER_FP16_SPATZ_PARAMS_BASE);

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

static int store_result(void *params, float16 *gather_result)
{
    volatile gather_fp16_spatz_params_t *gather_params;
    uint32_t shard_output_base;
    uint32_t axis_length;
    uint32_t start;
    uint32_t len;

    gather_params = (volatile gather_fp16_spatz_params_t *) params;
    shard_output_base = gather_params->shard_output;
    axis_length = gather_params->axis_length;
    start = gather_params->batch_start;
    len = gather_params->batch_len;

    uint32_t out_idx = 0;

    for (int b = 0; b < len; b++) {
        uint32_t global_batch = start + b;
        uint32_t global_idx_base = global_batch * axis_length;

        for (int i = 0; i < axis_length; i++) {
            uint32_t offset = out_idx * sizeof(float16);
            gather_result[global_idx_base + i] = mmio_fp16(shard_output_base + offset);
            out_idx++;
        }
    }

    return 0;
}

void MAGIA_gather_fp16_spatz(const float16 *data, float16 *output, uint32_t batch, uint32_t gather_dim_size, uint32_t axis_length, uint32_t index)
{
    int ret;
    volatile gather_fp16_spatz_params_t *params;

    params = (volatile gather_fp16_spatz_params_t *) GATHER_FP16_SPATZ_PARAMS_BASE;

    ret = init_input_params(params, data, batch, gather_dim_size, axis_length, index);
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
