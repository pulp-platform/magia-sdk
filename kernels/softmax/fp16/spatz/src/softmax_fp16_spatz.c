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

static int alloc_l1(void **params, uint32_t input_shape[4])
{
    volatile softmax_fp16_spatz_params_t *softmax_params;
    uintptr_t shard_input;
    uintptr_t shard_output;

    uint32_t total_rows;
    uint32_t row_len;
    uint32_t shard;
    uint32_t left;
    uint32_t r_start;
    uint32_t r_end;
    uint32_t r_len;

    total_rows = input_shape[0] * input_shape[1] * input_shape[2];
    row_len    = input_shape[3];

    shard = total_rows / NUM_HARTS;
    left  = total_rows % NUM_HARTS;

    r_start = HID * shard + (HID < left ? HID : left);
    r_end   = r_start + shard + (HID < left ? 1 : 0);
    r_len   = r_end - r_start;

    l1_alloc_init();

    softmax_params = l1_alloc(sizeof(softmax_fp16_spatz_params_t));
    if (!softmax_params)
        return ENOMEM;

    shard_input = l1_alloc(r_len * row_len * sizeof(float16));
    if (!shard_input)
        return ENOMEM;

    shard_output = l1_alloc(r_len * row_len * sizeof(float16));
    if (!shard_output)
        return ENOMEM;

    softmax_params->shard_input  = shard_input;
    softmax_params->shard_output = shard_output;
    softmax_params->r_start      = r_start;
    softmax_params->r_len        = r_len;
    softmax_params->w_len        = row_len;

    *params = (void *) softmax_params;

    return 0;
}

static int init_input_params(void *params, const float16 *input)
{
    volatile softmax_fp16_spatz_params_t *softmax_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t r_start;
    uint32_t r_len;
    uint32_t w_len;

    softmax_params = (volatile softmax_fp16_spatz_params_t *) params;
    r_start = softmax_params->r_start;
    r_len   = softmax_params->r_len;
    w_len   = softmax_params->w_len;

    if (r_len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's rows [r_start, r_start+r_len) are contiguous in L2. The Spatz task writes
       every output element, so shard_output is not zeroed here. */
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (input + r_start * w_len), (uint32_t) softmax_params->shard_input, r_len * w_len * sizeof(float16));
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
    uint32_t r_start;
    uint32_t r_len;
    uint32_t w_len;

    softmax_params = (volatile softmax_fp16_spatz_params_t *) params;
    r_start = softmax_params->r_start;
    r_len   = softmax_params->r_len;
    w_len   = softmax_params->w_len;

    if (r_len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's output rows [r_start, r_start+r_len) are contiguous in L2. */
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t) (output + r_start * w_len), (uint32_t) softmax_params->shard_output, r_len * w_len * sizeof(float16));
    eu_idma_wait_o2a(&eu_ctrl, WFE);

    return 0;
}

void MAGIA_softmax_fp16_spatz(const float16 *input, float16 *output, uint32_t input_shape[4])
{
    int ret;
    volatile softmax_fp16_spatz_params_t *params;

    ret = alloc_l1((void **)&params, input_shape);
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
