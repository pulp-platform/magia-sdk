#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"

#include "softmax_fp16_spatz.h"
#include "softmax_fp16_spatz_params.h"
#include "softmax_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "softmax_fp16_spatz"

static int alloc_l1(void **params, uint32_t outer_dim, uint32_t reduce_dim, uint32_t inner_dim)
{
    volatile softmax_fp16_spatz_params_t *softmax_params;
    uintptr_t shard_input;
    uintptr_t shard_output;

    uint32_t block;
    uint32_t shard;
    uint32_t left;
    uint32_t outer_start;
    uint32_t outer_end;
    uint32_t outer_len;

    block = reduce_dim * inner_dim;

    shard = outer_dim / NUM_HARTS;
    left  = outer_dim % NUM_HARTS;

    outer_start = HID * shard + (HID < left ? HID : left);
    outer_end   = outer_start + shard + (HID < left ? 1 : 0);
    outer_len   = outer_end - outer_start;

    l1_alloc_init();

    softmax_params = l1_alloc(sizeof(softmax_fp16_spatz_params_t));
    if (!softmax_params)
        return ENOMEM;

    shard_input = l1_alloc(outer_len * block * sizeof(float16));
    if (!shard_input)
        return ENOMEM;

    shard_output = l1_alloc(outer_len * block * sizeof(float16));
    if (!shard_output)
        return ENOMEM;

    softmax_params->shard_input  = shard_input;
    softmax_params->shard_output = shard_output;
    softmax_params->reduce_dim   = reduce_dim;
    softmax_params->inner_dim    = inner_dim;
    softmax_params->outer_start  = outer_start;
    softmax_params->outer_len    = outer_len;

    *params = (void *) softmax_params;

    return 0;
}

static int init_input_params(void *params, const float16 *input)
{
    volatile softmax_fp16_spatz_params_t *softmax_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t outer_start;
    uint32_t outer_len;
    uint32_t block;

    softmax_params = (volatile softmax_fp16_spatz_params_t *) params;
    outer_start = softmax_params->outer_start;
    outer_len   = softmax_params->outer_len;
    block       = softmax_params->reduce_dim * softmax_params->inner_dim;

    if (outer_len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's outer slices [outer_start, outer_start+outer_len) are contiguous in L2. The
       Spatz task writes every output element, so shard_output is not zeroed here. */
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (input + outer_start * block), (uint32_t) softmax_params->shard_input, outer_len * block * sizeof(float16));
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    int ret;

    eu_ctrl_init(&eu_ctrl);
    spatz_run_task_with_params(SOFTMAX_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *output)
{
    volatile softmax_fp16_spatz_params_t *softmax_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t outer_start;
    uint32_t outer_len;
    uint32_t block;

    softmax_params = (volatile softmax_fp16_spatz_params_t *) params;
    outer_start = softmax_params->outer_start;
    outer_len   = softmax_params->outer_len;
    block       = softmax_params->reduce_dim * softmax_params->inner_dim;

    if (outer_len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's output slices [outer_start, outer_start+outer_len) are contiguous in L2. */
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t) (output + outer_start * block), (uint32_t) softmax_params->shard_output, outer_len * block * sizeof(float16));
    eu_idma_wait_o2a(&eu_ctrl, WFE);

    return 0;
}

void MAGIA_softmax_fp16_spatz(const float16 *input, float16 *output, uint32_t outer_dim, uint32_t reduce_dim, uint32_t inner_dim)
{
    int ret;
    volatile softmax_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, outer_dim, reduce_dim, inner_dim);
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
