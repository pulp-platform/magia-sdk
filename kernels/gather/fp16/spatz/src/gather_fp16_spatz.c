#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"

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
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;

    uint32_t in_batch_stride;
    uint32_t batch_start;
    uint32_t batch_len;

    gather_params = (volatile gather_fp16_spatz_params_t *) params;
    in_batch_stride = gather_params->gather_dim_size * gather_params->axis_length;
    batch_start = gather_params->batch_start;
    batch_len   = gather_params->batch_len;

    if (batch_len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's batches [batch_start, batch_start+batch_len) are a contiguous L2 block.
       The Spatz task fully writes shard_output (axis_length per batch), so it is not zeroed
       here. */
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (data + batch_start * in_batch_stride), (uint32_t) gather_params->shard_input, batch_len * in_batch_stride * sizeof(float16));
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    int ret;

    eu_ctrl_init(&eu_ctrl);

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
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t axis_length;
    uint32_t start;
    uint32_t len;

    gather_params = (volatile gather_fp16_spatz_params_t *) params;
    axis_length = gather_params->axis_length;
    start = gather_params->batch_start;
    len = gather_params->batch_len;

    if (len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's output batches [batch_start, batch_start+batch_len) are contiguous in L2. */
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t) (gather_result + start * axis_length), (uint32_t) gather_params->shard_output, len * axis_length * sizeof(float16));
    eu_idma_wait_o2a(&eu_ctrl, WFE);

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
