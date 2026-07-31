#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "kernels_dma_utils.h"

#include "globalaveragepool_fp16_spatz.h"
#include "globalaveragepool_fp16_spatz_params.h"
#include "globalaveragepool_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "globalaveragepool_fp16_spatz"

static int alloc_l1(void **params, uint32_t input_shape[4])
{
    volatile globalaveragepool_fp16_spatz_params_t *gap_params;
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

    gap_params = l1_alloc(sizeof(globalaveragepool_fp16_spatz_params_t));
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

/* The shard is a run of whole (batch, channel) planes, so it is contiguous in X and
 * lands contiguously in L1: one transfer. shard_Y is not pre-cleared, the task writes
 * one average per plane it owns. */
static int init_input_params(kdma_t *d, void *params, const float16 *X)
{
    volatile globalaveragepool_fp16_spatz_params_t *gap_params;

    gap_params = (volatile globalaveragepool_fp16_spatz_params_t *) params;

    kdma_in(d,
            (uintptr_t) (X + (uintptr_t) gap_params->start * gap_params->hw_len),
            gap_params->shard_X,
            gap_params->len * gap_params->hw_len * sizeof(float16));

    return 0;
}

static int offload_spatz_task(kdma_t *d, void *params)
{
    int ret;

    spatz_run_task_with_params(GLOBALAVERAGEPOOL_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(&d->eu, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(kdma_t *d, void *params, float16 *Y)
{
    volatile globalaveragepool_fp16_spatz_params_t *gap_params;

    gap_params = (volatile globalaveragepool_fp16_spatz_params_t *) params;

    kdma_out(d,
             (uintptr_t) (Y + gap_params->start),
             gap_params->shard_Y,
             gap_params->len * sizeof(float16));

    return 0;
}

void MAGIA_globalaveragepool_fp16_spatz(const float16 *X, float16 *Y, uint32_t input_shape[4])
{
    int ret;
    volatile globalaveragepool_fp16_spatz_params_t *params;
    kdma_t d;

    kdma_open(&d);

    ret = alloc_l1((void **)&params, input_shape);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params(&d, (void *)params, X);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task(&d, (void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result(&d, (void *)params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
