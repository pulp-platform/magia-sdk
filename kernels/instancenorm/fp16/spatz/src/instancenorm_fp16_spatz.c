#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"

#include "instancenorm_fp16_spatz.h"
#include "instancenorm_fp16_spatz_params.h"
#include "instancenorm_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "instancenorm_fp16_spatz"

static int alloc_l1(void **params, uint32_t input_shape[4], uint32_t *out_num_channels)
{
    volatile instancenorm_fp16_spatz_params_t *instancenorm_params;
    uintptr_t shard_input;
    uintptr_t shard_output;
    uintptr_t gamma;
    uintptr_t beta;
    uintptr_t eps;

    uint32_t total_instances;
    uint32_t instance_len;
    uint32_t num_channels;
    uint32_t shard;
    uint32_t left;
    uint32_t inst_start;
    uint32_t inst_end;
    uint32_t inst_len;

    num_channels = input_shape[1];
    total_instances = input_shape[0] * input_shape[1];
    instance_len = input_shape[2] * input_shape[3];

    shard = total_instances / NUM_HARTS;
    left  = total_instances % NUM_HARTS;

    inst_start = HID * shard + (HID < left ? HID : left);
    inst_end   = inst_start + shard + (HID < left ? 1 : 0);
    inst_len   = inst_end - inst_start;

    l1_alloc_init();

    instancenorm_params = l1_alloc(sizeof(instancenorm_fp16_spatz_params_t));
    if (!instancenorm_params)
        return ENOMEM;

    shard_input = l1_alloc(inst_len * instance_len * sizeof(float16));
    if (!shard_input)
        return ENOMEM;

    shard_output = l1_alloc(inst_len * instance_len * sizeof(float16));
    if (!shard_output)
        return ENOMEM;

    gamma = l1_alloc(num_channels * sizeof(float16));
    if (!gamma)
        return ENOMEM;

    beta = l1_alloc(num_channels * sizeof(float16));
    if (!beta)
        return ENOMEM;

    eps = l1_alloc(sizeof(float16));
    if (!eps)
        return ENOMEM;

    instancenorm_params->shard_input  = shard_input;
    instancenorm_params->shard_output = shard_output;
    instancenorm_params->gamma        = gamma;
    instancenorm_params->beta         = beta;
    instancenorm_params->eps          = eps;
    instancenorm_params->inst_start   = inst_start;
    instancenorm_params->inst_len     = inst_len;
    instancenorm_params->hw_len       = instance_len;
    instancenorm_params->num_channels = num_channels;

    *params = (void *) instancenorm_params;
    *out_num_channels = num_channels;

    return 0;
}

static int init_input_params(void *params, const float16 *input, const float16 *scale, const float16 *B, const float16 epsilon)
{
    volatile instancenorm_fp16_spatz_params_t *instancenorm_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t num_channels;
    uint32_t inst_start;
    uint32_t inst_len;
    uint32_t hw_len;

    instancenorm_params = (volatile instancenorm_fp16_spatz_params_t *) params;
    inst_start   = instancenorm_params->inst_start;
    inst_len     = instancenorm_params->inst_len;
    hw_len       = instancenorm_params->hw_len;
    num_channels = instancenorm_params->num_channels;

    mmio_fp16(instancenorm_params->eps) = epsilon;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* Per-channel gamma/beta (full num_channels, contiguous) are needed by every tile. */
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) scale, (uint32_t) instancenorm_params->gamma, num_channels * sizeof(float16));
    eu_idma_wait_a2o(&eu_ctrl, WFE);
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) B, (uint32_t) instancenorm_params->beta, num_channels * sizeof(float16));
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    if (inst_len == 0)
        return 0;

    /* This tile's instances [inst_start, inst_end) are contiguous in L2. The Spatz task fully
       writes shard_output, so it is not zeroed here. */
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (input + inst_start * hw_len), (uint32_t) instancenorm_params->shard_input, inst_len * hw_len * sizeof(float16));
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    int ret;

    eu_ctrl_init(&eu_ctrl);
    spatz_run_task_with_params(INSTANCENORM_FP16_SPATZ_TASK, (uint32_t)params);

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
    volatile instancenorm_fp16_spatz_params_t *instancenorm_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t inst_start;
    uint32_t inst_len;
    uint32_t hw_len;

    instancenorm_params = (volatile instancenorm_fp16_spatz_params_t *) params;
    inst_start = instancenorm_params->inst_start;
    inst_len   = instancenorm_params->inst_len;
    hw_len     = instancenorm_params->hw_len;

    if (inst_len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's output instances [inst_start, inst_end) are contiguous in L2. */
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t) (output + inst_start * hw_len), (uint32_t) instancenorm_params->shard_output, inst_len * hw_len * sizeof(float16));
    eu_idma_wait_o2a(&eu_ctrl, WFE);

    return 0;
}

void MAGIA_instancenorm_fp16_spatz(const float16 *input, float16 *output, const float16 *scale, const float16 *B, const float16 epsilon, uint32_t input_shape[4])
{
    int ret;
    volatile instancenorm_fp16_spatz_params_t *params;
    uint32_t num_channels;

    ret = alloc_l1((void **)&params, input_shape, &num_channels);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params((void *)params, input, scale, B, epsilon);
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
