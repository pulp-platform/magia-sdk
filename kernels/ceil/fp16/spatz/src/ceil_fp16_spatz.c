#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"

#include "ceil_fp16_spatz.h"
#include "ceil_fp16_spatz_params.h"
#include "ceil_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "ceil_fp16_spatz"

static int alloc_l1(void **params, size_t size)
{
    volatile ceil_fp16_spatz_params_t *ceil_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;

    size_t shard_start;
    size_t shard_end;
    size_t shard_len;
    size_t elems;
    size_t left;

    elems = size / NUM_HARTS;
    left = size % NUM_HARTS;
    shard_start = HID * elems + (HID < left ? HID : left);
    shard_end = shard_start + elems + (HID < left ? 1 : 0);
    shard_len = shard_end - shard_start;

    l1_alloc_init();

    ceil_params = l1_alloc(sizeof(ceil_fp16_spatz_params_t));
    if (!ceil_params)
        return ENOMEM;

    shard_X = l1_alloc(shard_len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(shard_len * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    ceil_params->shard_X = shard_X;
    ceil_params->shard_Y = shard_Y;
    ceil_params->start = shard_start;
    ceil_params->len = shard_len;
    ceil_params->end = shard_end;

    *params = (void *) ceil_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile ceil_fp16_spatz_params_t *ceil_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t start;
    uint32_t len;

    ceil_params = (volatile ceil_fp16_spatz_params_t *) params;
    start = ceil_params->start;
    len = ceil_params->len;

    if (len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's slice [start, start+len) is contiguous in L2. The Spatz task writes every
       output (Y = ceil(X)), so shard_Y is not zeroed here. */
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (X + start), (uint32_t) ceil_params->shard_X, len * sizeof(float16));
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    return 0;
}


static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    int ret;

    eu_ctrl_init(&eu_ctrl);
    spatz_run_task_with_params(CEIL_FP16_SPATZ_TASK, params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void* params, float16 *dst)
{
    volatile ceil_fp16_spatz_params_t *ceil_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t start;
    uint32_t len;

    ceil_params = (volatile ceil_fp16_spatz_params_t *) params;
    start = ceil_params->start;
    len = ceil_params->len;

    if (len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's output slice [start, start+len) is contiguous in L2. */
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t) (dst + start), (uint32_t) ceil_params->shard_Y, len * sizeof(float16));
    eu_idma_wait_o2a(&eu_ctrl, WFE);

    return 0;
}


void MAGIA_ceil_fp16_spatz(const float16 *X, float16 *Y, uint32_t size)
{
    int ret;
    volatile ceil_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, size);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params(params, X);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task(params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result(params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
