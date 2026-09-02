#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"

#include "exp_fp16_spatz.h"
#include "exp_fp16_spatz_params.h"
#include "exp_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "exp_fp16_spatz"

static int alloc_l1(void **params, uint32_t size)
{
    volatile exp_fp16_spatz_params_t *exp_params;
    uintptr_t shard_input;
    uintptr_t shard_output;

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

    exp_params = l1_alloc(sizeof(exp_fp16_spatz_params_t));
    if (!exp_params)
        return ENOMEM;

    shard_input = l1_alloc(len * sizeof(float16));
    if (!shard_input)
        return ENOMEM;

    shard_output = l1_alloc(len * sizeof(float16));
    if (!shard_output)
        return ENOMEM;

    exp_params->shard_input  = shard_input;
    exp_params->shard_output = shard_output;
    exp_params->start        = start;
    exp_params->len          = len;
    exp_params->end          = end;

    *params = (void *) exp_params;

    return 0;
}

static int init_input_params(void *params, const float16 *input)
{
    volatile exp_fp16_spatz_params_t *exp_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t start;
    uint32_t len;

    exp_params = (volatile exp_fp16_spatz_params_t *) params;
    start = exp_params->start;
    len   = exp_params->len;

    if (len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's slice [start, start+len) is contiguous in L2. The Spatz task writes every
       output (Y = exp(X)), so shard_output is not zeroed here. */
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (input + start), (uint32_t) exp_params->shard_input, len * sizeof(float16));
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    int ret;

    eu_ctrl_init(&eu_ctrl);
    spatz_run_task_with_params(EXP_FP16_SPATZ_TASK, (uint32_t)params);

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
    volatile exp_fp16_spatz_params_t *exp_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t start;
    uint32_t len;

    exp_params = (volatile exp_fp16_spatz_params_t *) params;
    start = exp_params->start;
    len = exp_params->len;

    if (len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's output slice [start, start+len) is contiguous in L2. */
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t) (dst + start), (uint32_t) exp_params->shard_output, len * sizeof(float16));
    eu_idma_wait_o2a(&eu_ctrl, WFE);

    return 0;
}

void MAGIA_exp_fp16_spatz(const float16 *input, float16 *output, uint32_t size)
{
    int ret;
    volatile exp_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, size);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, input);
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
