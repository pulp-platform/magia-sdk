#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "instancenorm_fp16_spatz.h"
#include "instancenorm_fp16_spatz_mem_layout.h"
#include "instancenorm_fp16_spatz_params.h"
#include "instancenorm_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int init_input_params(void *params, const float16 *input, const float16 *scale, const float16 *B, const float16 epsilon)
{
    volatile instancenorm_fp16_spatz_params_t *instancenorm_params;
    uint32_t shard;
    uint32_t left;
    uint32_t inst_start;
    uint32_t inst_end;
    uint32_t inst_len;
    uint32_t hw_len;
    uint32_t local_idx;

    instancenorm_params = (volatile instancenorm_fp16_spatz_params_t *) params;

    shard = TOTAL_INSTANCES / NUM_HARTS;
    left  = TOTAL_INSTANCES % NUM_HARTS;

    inst_start = HID * shard + (HID < left ? HID : left);
    inst_end   = inst_start + shard + (HID < left ? 1 : 0);
    inst_len   = inst_end - inst_start;
    hw_len     = INSTANCE_LEN;

    local_idx = 0;
    for (int inst = inst_start; inst < inst_end; inst++) {
        uint32_t global_base = inst * hw_len;

        for (int i = 0; i < hw_len; i++) {
            uint32_t global_idx = global_base + i;
            uint32_t offset = local_idx * sizeof(float16);

            mmio_fp16(SHARD_INPUT_BASE + offset) = input[global_idx];
            mmio_fp16(SHARD_OUTPUT_BASE + offset) = 0;

            local_idx++;
        }
    }

    for (int i = 0; i < INPUT0_DIM1; i++) {
        uint32_t offset = i * sizeof(float16);

        mmio_fp16(GAMMA_BASE + offset) = scale[i];
        mmio_fp16(BETA_BASE + offset) = B[i];
    }

    mmio_fp16(EPS_BASE) = epsilon;

    instancenorm_params->shard_input = SHARD_INPUT_BASE;
    instancenorm_params->shard_output = SHARD_OUTPUT_BASE;
    instancenorm_params->gamma = GAMMA_BASE;
    instancenorm_params->beta  = BETA_BASE;
    instancenorm_params->eps   = EPS_BASE;
    instancenorm_params->inst_start = inst_start;
    instancenorm_params->inst_len   = inst_len;
    instancenorm_params->hw_len    = hw_len;
    instancenorm_params->num_channels = INPUT0_DIM1;

    return 0;
}

static int offload_spatz_task()
{
    eu_controller_t eu_ctrl;
    eu_config_t eu_cfg;
    int ret;

    eu_cfg.hartid = HID;
    eu_ctrl.base = NULL;
    eu_ctrl.cfg = &eu_cfg;
    eu_ctrl.api = &eu_api;

    spatz_init(SPATZ_BINARY_START);
    spatz_run_task_with_params(INSTANCENORM_FP16_SPATZ_TASK, INSTANCENORM_FP16_SPATZ_PARAMS_BASE);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] Wait on Spatz task completion failed with error: %d\n", HID, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();
    spatz_clk_dis();

exit:
    return ret;
}

static int store_result(void *params, float16 *output)
{
    volatile instancenorm_fp16_spatz_params_t *instancenorm_params;
    uint32_t shard_output_base;
    uint32_t local_idx;

    instancenorm_params = (volatile instancenorm_fp16_spatz_params_t *) params;
    shard_output_base = instancenorm_params->shard_output;
    local_idx = 0;

    /* Scrittura indietro da L1 a L2 */
    for (int inst = 0; inst < instancenorm_params->inst_len; inst++) {
        uint32_t global_inst_idx = instancenorm_params->inst_start + inst;
        uint32_t global_base = global_inst_idx * instancenorm_params->hw_len;

        for (int i = 0; i < instancenorm_params->hw_len; i++) {
            uint32_t global_idx = global_base + i;
            uint32_t offset = local_idx * sizeof(float16);

            output[global_idx] = mmio_fp16(shard_output_base + offset);

            local_idx++;
        }
    }

    return 0;
}

void MAGIA_instancenorm_fp16_spatz(const float16 *input, float16 *output, const float16 *scale, const float16 *B, const float16 epsilon)
{
    int ret;
    volatile instancenorm_fp16_spatz_params_t *params;

    params = (volatile instancenorm_fp16_spatz_params_t *) INSTANCENORM_FP16_SPATZ_PARAMS_BASE;

    ret = init_input_params(params, input, scale, B, epsilon);
    if (ret != 0) {
        printf("[CV32 (%d)] Params initialization failed with error: %d\n", HID, ret);
        return;
    }

    ret = offload_spatz_task();
    if (ret != 0) {
        printf("[CV32 (%d)] Spatz task offloading failed with error: %d\n", HID, ret);
        return;
    }

    ret = store_result(params, output);
    if (ret != 0) {
        printf("[CV32 (%d)] Result write back failed with error: %d\n", HID, ret);
    }

}
