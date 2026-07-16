#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "gather_fp16_spatz.h"
#include "gather_fp16_spatz_params.h"
#include "gather_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "gather_fp16_spatz"

static int alloc_l1(void **params, uint32_t in_shape[4], uint32_t batch, uint32_t gather_dim_size, uint32_t axis_length, uint32_t index, uint32_t axis)
{
    volatile gather_fp16_spatz_params_t *gather_params;
    uintptr_t shard_input;
    uintptr_t shard_output;

    uint32_t shard_iteration;
    uint32_t batch_start;
    uint32_t batch_len;
    uint32_t in_batch_stride;

    if (axis == 1) {
        shard_iteration = (in_shape[0] + NUM_HARTS - 1) / NUM_HARTS;
    } else if (axis == 2) {
        shard_iteration = (in_shape[0] * in_shape[1] + NUM_HARTS - 1) / NUM_HARTS;
    } else if (axis == 3) {
        shard_iteration = (in_shape[0] * in_shape[1] * in_shape[2] + NUM_HARTS - 1) / NUM_HARTS;
    } else {
        return EINVAL;
    }

    batch_start = HID * shard_iteration;
    batch_len   = shard_iteration;

    in_batch_stride = gather_dim_size * axis_length;

    l1_alloc_init();

    gather_params = l1_alloc(sizeof(gather_fp16_spatz_params_t));
    if (!gather_params)
        return ENOMEM;

    shard_input = l1_alloc(batch_len * in_batch_stride * sizeof(float16));
    if (!shard_input)
        return ENOMEM;

    shard_output = l1_alloc(batch_len * axis_length * sizeof(float16));
    if (!shard_output)
        return ENOMEM;

    gather_params->shard_input    = shard_input;
    gather_params->shard_output   = shard_output;
    gather_params->gather_dim_size = gather_dim_size;
    gather_params->axis_length    = axis_length;
    gather_params->batch_start    = batch_start;
    gather_params->batch_len      = batch_len;
    gather_params->index          = index;

    *params = (void *) gather_params;

    return 0;
}

static int init_input_params(void *params, const float16 *data)
{
    volatile gather_fp16_spatz_params_t *gather_params;
    uintptr_t shard_input_base;
    uintptr_t shard_output_base;
    uint32_t batch_start;
    uint32_t batch_end;
    uint32_t in_batch_stride;
    uint32_t l1_in_idx;
    uint32_t l1_out_idx;

    gather_params = (volatile gather_fp16_spatz_params_t *) params;
    shard_input_base  = gather_params->shard_input;
    shard_output_base = gather_params->shard_output;

    batch_start = gather_params->batch_start;
    batch_end   = batch_start + gather_params->batch_len;
    in_batch_stride = gather_params->gather_dim_size * gather_params->axis_length;

    l1_in_idx = 0;
    l1_out_idx = 0;

    for (uint32_t b = batch_start; b < batch_end; b++) {
        uint32_t global_idx_in = b * in_batch_stride;
        uint32_t offset;

        for (uint32_t i = 0; i < in_batch_stride; i++) {
            offset = l1_in_idx * sizeof(float16);
            mmio_fp16(shard_input_base + offset) = data[global_idx_in + i];
            l1_in_idx++;
        }

        for (uint32_t i = 0; i < gather_params->axis_length; i++) {
            offset = l1_out_idx * sizeof(float16);
            mmio_fp16(shard_output_base + offset) = 0;
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

    spatz_run_task_with_params(GATHER_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *gather_result)
{
    volatile gather_fp16_spatz_params_t *gather_params;
    uintptr_t shard_output_base;
    uint32_t axis_length;
    uint32_t start;
    uint32_t len;
    uint32_t out_idx;

    gather_params = (volatile gather_fp16_spatz_params_t *) params;
    shard_output_base = gather_params->shard_output;
    axis_length = gather_params->axis_length;
    start = gather_params->batch_start;
    len = gather_params->batch_len;

    out_idx = 0;

    for (uint32_t b = 0; b < len; b++) {
        uint32_t global_batch = start + b;
        uint32_t global_idx_base = global_batch * axis_length;

        for (uint32_t i = 0; i < axis_length; i++) {
            uint32_t offset = out_idx * sizeof(float16);
            gather_result[global_idx_base + i] = mmio_fp16(shard_output_base + offset);
            out_idx++;
        }
    }

    return 0;
}

void MAGIA_gather_fp16_spatz(const float16 *data, uint32_t in_shape[4], float16 *output, uint32_t batch, uint32_t gather_dim_size, uint32_t axis_length, uint32_t index, uint32_t axis)
{
    int ret;
    volatile gather_fp16_spatz_params_t *params;

    if (axis == 0) {
        printf("[CV32 (%d)] [%s] Axis 0 gather is not supported on Spatz kernel! Handle via flat L2 DMA copy.\n", HID, KERNEL_NAME);
        return;
    }

    ret = alloc_l1((void **)&params, in_shape, batch, gather_dim_size, axis_length, index, axis);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, data);
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
