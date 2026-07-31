#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernels_dma_utils.h"

#include "sigmoid_fp16_spatz.h"
#include "sigmoid_fp16_spatz_params.h"
#include "sigmoid_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "sigmoid_fp16_spatz"

static int alloc_l1(void **params, uint32_t size)
{
    volatile sigmoid_fp16_spatz_params_t *sigmoid_params;
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

    sigmoid_params = l1_alloc(sizeof(sigmoid_fp16_spatz_params_t));
    if (!sigmoid_params)
        return ENOMEM;

    shard_X = l1_alloc(len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(len * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    sigmoid_params->shard_X = shard_X;
    sigmoid_params->shard_Y = shard_Y;
    sigmoid_params->start   = start;
    sigmoid_params->len     = len;
    sigmoid_params->end     = end;

    *params = (void *) sigmoid_params;

    return 0;
}

/* The shard is a contiguous element range, so one transfer covers it. No shard_Y prefill:
 * the task writes every element of the shard. */
static int init_input_params(void *params, kdma_t *d, const float16 *X)
{
    volatile sigmoid_fp16_spatz_params_t *sigmoid_params;

    sigmoid_params = (volatile sigmoid_fp16_spatz_params_t *) params;

    kdma_in(d,
            (uintptr_t)(X + sigmoid_params->start),
            sigmoid_params->shard_X,
            sigmoid_params->len * sizeof(float16));

    return 0;
}

static int offload_spatz_task(kdma_t *d, void *params)
{
    int ret;

    spatz_run_task_with_params(SIGMOID_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(&d->eu, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, kdma_t *d, float16 *dst)
{
    volatile sigmoid_fp16_spatz_params_t *sigmoid_params;

    sigmoid_params = (volatile sigmoid_fp16_spatz_params_t *) params;

    kdma_out(d,
             (uintptr_t)(dst + sigmoid_params->start),
             sigmoid_params->shard_Y,
             sigmoid_params->len * sizeof(float16));

    return 0;
}

void MAGIA_sigmoid_fp16_spatz(const float16 *X, float16 *Y, uint32_t size)
{
    int ret;
    volatile sigmoid_fp16_spatz_params_t *params;
    kdma_t d;

    ret = alloc_l1((void **)&params, size);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    if (params->len == 0)
        return;

    kdma_open(&d);

    ret = init_input_params((void *)params, &d, X);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task(&d, (void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, &d, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
