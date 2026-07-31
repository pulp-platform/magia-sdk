#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "kernels_dma_utils.h"

#include "relu_fp16_spatz.h"
#include "relu_fp16_spatz_params.h"
#include "relu_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "relu_fp16_spatz"

static int alloc_l1(void **params, uint32_t size)
{
    volatile relu_fp16_spatz_params_t *relu_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;

    uint32_t shard;
    uint32_t left;
    uint32_t start;
    uint32_t end;
    uint32_t len;

    shard = size / NUM_HARTS;
    left  = size % NUM_HARTS;
    start = HID * shard + (HID < left ? HID : left);
    end   = start + shard + (HID < left ? 1 : 0);
    len   = end - start;

    l1_alloc_init();

    relu_params = l1_alloc(sizeof(relu_fp16_spatz_params_t));
    if (!relu_params)
        return ENOMEM;

    shard_X = l1_alloc(len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(len * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    relu_params->shard_X = shard_X;
    relu_params->shard_Y = shard_Y;
    relu_params->start   = start;
    relu_params->len     = len;
    relu_params->end     = end;

    *params = (void *) relu_params;

    return 0;
}

/* This tile's shard is a contiguous range of X, so one transfer brings it in. shard_Y
 * is not pre-cleared: the task writes every element of it. */
static int init_input_params(kdma_t *d, void *params, const float16 *X)
{
    volatile relu_fp16_spatz_params_t *relu_params;

    relu_params = (volatile relu_fp16_spatz_params_t *) params;

    kdma_in(d,
            (uintptr_t) (X + relu_params->start),
            relu_params->shard_X,
            relu_params->len * sizeof(float16));

    return 0;
}

static int offload_spatz_task(kdma_t *d, void *params)
{
    int ret;

    spatz_run_task_with_params(RELU_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(&d->eu, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(kdma_t *d, void *params, float16 *dst)
{
    volatile relu_fp16_spatz_params_t *relu_params;

    relu_params = (volatile relu_fp16_spatz_params_t *) params;

    kdma_out(d,
             (uintptr_t) (dst + relu_params->start),
             relu_params->shard_Y,
             relu_params->len * sizeof(float16));

    return 0;
}

void MAGIA_relu_fp16_spatz(const float16 *X, float16 *Y, uint32_t size)
{
    int ret;
    volatile relu_fp16_spatz_params_t *params;
    kdma_t d;

    kdma_open(&d);

    ret = alloc_l1((void **)&params, size);
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
