#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"

#include "leakyrelu_fp16_spatz.h"
#include "leakyrelu_fp16_spatz_params.h"
#include "leakyrelu_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "leakyrelu_fp16_spatz"

static int allocate_l1(void **params, uint32_t size)
{
    volatile leakyrelu_fp16_spatz_params_t *leakyrelu_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;
    uintptr_t alpha;

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

    leakyrelu_params = l1_alloc(sizeof(leakyrelu_fp16_spatz_params_t));
    if (!leakyrelu_params)
        return ENOMEM;

    shard_X = l1_alloc(shard_len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(shard_len * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    alpha = l1_alloc(sizeof(float16));
    if (!alpha)
        return ENOMEM;

    leakyrelu_params->shard_X = shard_X;
    leakyrelu_params->shard_Y = shard_Y;
    leakyrelu_params->alpha = alpha;
    leakyrelu_params->start = shard_start;
    leakyrelu_params->end = shard_end;
    leakyrelu_params->len = shard_len;

    *params = (void *) leakyrelu_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, float16 a)
{
    volatile leakyrelu_fp16_spatz_params_t *leakyrelu_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t start;
    uint32_t len;

    leakyrelu_params = (volatile leakyrelu_fp16_spatz_params_t *) params;
    start = leakyrelu_params->start;
    len = leakyrelu_params->len;

    mmio_fp16(leakyrelu_params->alpha) = a;

    if (len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's slice [start, start+len) is contiguous in L2. The Spatz task writes every
       output (Y = leakyrelu(X, alpha)), so shard_Y is not zeroed here. */
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (X + start), (uint32_t) leakyrelu_params->shard_X, len * sizeof(float16));
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    int ret;

    eu_ctrl_init(&eu_ctrl);
    spatz_run_task_with_params(LEAKYRELU_FP16_SPATZ_TASK, params);

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
    volatile leakyrelu_fp16_spatz_params_t *leakyrelu_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t start;
    uint32_t len;

    leakyrelu_params = (volatile leakyrelu_fp16_spatz_params_t *) params;
    start = leakyrelu_params->start;
    len = leakyrelu_params->len;

    if (len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's output slice [start, start+len) is contiguous in L2. */
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t) (dst + start), (uint32_t) leakyrelu_params->shard_Y, len * sizeof(float16));
    eu_idma_wait_o2a(&eu_ctrl, WFE);

    return 0;
}

void MAGIA_leakyrelu_fp16_spatz(const float16 *X, float16 *Y, uint32_t size, float16 alpha)
{
    int ret;
    volatile leakyrelu_fp16_spatz_params_t *params;

    ret = allocate_l1(&params, size);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params(params, X, alpha);
    if (ret != 0) {
        printf("[CV32 (%d) Params initialization failed with error: %d\n]", HID, KERNEL_NAME, ret);
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
