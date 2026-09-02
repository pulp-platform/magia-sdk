#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"

#include "div_fp16_spatz.h"
#include "div_fp16_spatz_params.h"
#include "div_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "div_fp16_spatz"

static int alloc_l1(void **params, uint32_t size)
{
    volatile div_fp16_spatz_params_t *div_params;
    uintptr_t shard_A;
    uintptr_t shard_B;
    uintptr_t shard_C;

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

    div_params = l1_alloc(sizeof(div_fp16_spatz_params_t));
    if (!div_params)
        return ENOMEM;

    shard_A = l1_alloc(len * sizeof(float16));
    if (!shard_A)
        return ENOMEM;

    shard_B = l1_alloc(len * sizeof(float16));
    if (!shard_B)
        return ENOMEM;

    shard_C = l1_alloc(len * sizeof(float16));
    if (!shard_C)
        return ENOMEM;

    div_params->shard_A = shard_A;
    div_params->shard_B = shard_B;
    div_params->shard_C = shard_C;
    div_params->start   = start;
    div_params->len     = len;
    div_params->end     = end;

    *params = (void *) div_params;

    return 0;
}

static int init_input_params(void *params, const float16 *A, const float16 *B)
{
    volatile div_fp16_spatz_params_t *div_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t start;
    uint32_t len;

    div_params = (volatile div_fp16_spatz_params_t *) params;
    start = div_params->start;
    len   = div_params->len;

    if (len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's slice [start, start+len) is contiguous in both operands. The Spatz task
       writes every output (C = A / B), so shard_C is not zeroed here. */
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (A + start), (uint32_t) div_params->shard_A, len * sizeof(float16));
    eu_idma_wait_a2o(&eu_ctrl, WFE);
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (B + start), (uint32_t) div_params->shard_B, len * sizeof(float16));
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    int ret;

    eu_ctrl_init(&eu_ctrl);
    spatz_run_task_with_params(DIV_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *dst)
{
    volatile div_fp16_spatz_params_t *div_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t start;
    uint32_t len;

    div_params = (volatile div_fp16_spatz_params_t *) params;
    start = div_params->start;
    len = div_params->len;

    if (len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's output slice [start, start+len) is contiguous in L2. */
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t) (dst + start), (uint32_t) div_params->shard_C, len * sizeof(float16));
    eu_idma_wait_o2a(&eu_ctrl, WFE);

    return 0;
}

void MAGIA_div_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t size)
{
    int ret;
    volatile div_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, size);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, A, B);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task((void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result((void *)params, C);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
