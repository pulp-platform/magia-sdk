#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"

#include "globalmaxpool_fp16_spatz.h"
#include "globalmaxpool_fp16_spatz_params.h"
#include "globalmaxpool_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "globalmaxpool_fp16_spatz"

static int alloc_l1(void **params, uint32_t input_shape[4])
{
    volatile globalmaxpool_fp16_spatz_params_t *gap_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;

    uint32_t channel_num;
    uint32_t channel_len;
    uint32_t shard;
    uint32_t left;
    uint32_t start;
    uint32_t end;
    uint32_t len;

    channel_num = input_shape[0] * input_shape[1];
    channel_len = input_shape[2] * input_shape[3];

    shard = channel_num / NUM_HARTS;
    left  = channel_num % NUM_HARTS;

    start = HID * shard + (HID < left ? HID : left);
    end = start + shard + (HID < left ? 1 : 0);
    len = end - start;

    l1_alloc_init();

    gap_params = l1_alloc(sizeof(globalmaxpool_fp16_spatz_params_t));
    if (!gap_params)
        return ENOMEM;

    shard_X = l1_alloc(len * channel_len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(len * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    gap_params->shard_X = shard_X;
    gap_params->shard_Y = shard_Y;
    gap_params->hw_len  = channel_len;
    gap_params->start   = start;
    gap_params->len     = len;

    *params = (void *) gap_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile globalmaxpool_fp16_spatz_params_t *gap_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t start;
    uint32_t len;
    uint32_t hw_len;

    gap_params = (volatile globalmaxpool_fp16_spatz_params_t *) params;
    start  = gap_params->start;
    len    = gap_params->len;
    hw_len = gap_params->hw_len;

    if (len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's channels [start, start+len) are contiguous in L2. The Spatz task writes
       every output (one max per channel), so shard_Y is not zeroed here. */
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (X + start * hw_len), (uint32_t) gap_params->shard_X, len * hw_len * sizeof(float16));
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    int ret;

    eu_ctrl_init(&eu_ctrl);
    spatz_run_task_with_params(GLOBALMAXPOOL_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *Y)
{
    volatile globalmaxpool_fp16_spatz_params_t *gap_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t start;
    uint32_t len;

    gap_params = (volatile globalmaxpool_fp16_spatz_params_t *) params;
    start = gap_params->start;
    len   = gap_params->len;

    if (len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's outputs [start, start+len) are contiguous in L2. */
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t) (Y + start), (uint32_t) gap_params->shard_Y, len * sizeof(float16));
    eu_idma_wait_o2a(&eu_ctrl, WFE);

    return 0;
}

void MAGIA_globalmaxpool_fp16_spatz(const float16 *X, float16 *Y, uint32_t input_shape[4])
{
    int ret;
    volatile globalmaxpool_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, input_shape);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, X);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task((void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
