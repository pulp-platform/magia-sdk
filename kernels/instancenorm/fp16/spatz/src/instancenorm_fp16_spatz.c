#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

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
    uintptr_t shard_input_base;
    uintptr_t shard_output_base;
    uintptr_t gamma_base;
    uintptr_t beta_base;
    uintptr_t eps_base;
    uint32_t local_idx;
    uint32_t inst_end;

    instancenorm_params = (volatile instancenorm_fp16_spatz_params_t *) params;
    shard_input_base  = instancenorm_params->shard_input;
    shard_output_base = instancenorm_params->shard_output;
    gamma_base        = instancenorm_params->gamma;
    beta_base         = instancenorm_params->beta;
    eps_base          = instancenorm_params->eps;
    inst_end          = instancenorm_params->inst_start + instancenorm_params->inst_len;

    local_idx = 0;
    for (uint32_t inst = instancenorm_params->inst_start; inst < inst_end; inst++) {
        uint32_t global_base = inst * instancenorm_params->hw_len;

        for (uint32_t i = 0; i < instancenorm_params->hw_len; i++) {
            uint32_t global_idx = global_base + i;
            uint32_t offset = local_idx * sizeof(float16);

            mmio_fp16(shard_input_base + offset) = input[global_idx];
            mmio_fp16(shard_output_base + offset) = 0;

            local_idx++;
        }
    }

    for (uint32_t i = 0; i < instancenorm_params->num_channels; i++) {
        uint32_t offset = i * sizeof(float16);

        mmio_fp16(gamma_base + offset) = scale[i];
        mmio_fp16(beta_base + offset) = B[i];
    }

    mmio_fp16(eps_base) = epsilon;

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    eu_config_t eu_cfg;
    int ret;

    eu_cfg.hartid = HID;
    eu_ctrl.base = NULL;
    eu_ctrl.cfg = &eu_cfg;
    eu_ctrl.api = &eu_api;

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
    uintptr_t shard_output_base;
    uint32_t local_idx;

    instancenorm_params = (volatile instancenorm_fp16_spatz_params_t *) params;
    shard_output_base = instancenorm_params->shard_output;
    local_idx = 0;

    for (uint32_t inst = 0; inst < instancenorm_params->inst_len; inst++) {
        uint32_t global_inst_idx = instancenorm_params->inst_start + inst;
        uint32_t global_base = global_inst_idx * instancenorm_params->hw_len;

        for (uint32_t i = 0; i < instancenorm_params->hw_len; i++) {
            uint32_t global_idx = global_base + i;
            uint32_t offset = local_idx * sizeof(float16);

            output[global_idx] = mmio_fp16(shard_output_base + offset);

            local_idx++;
        }
    }

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
