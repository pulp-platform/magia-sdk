#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "batchnorm_fp16_spatz.h"
#include "batchnorm_fp16_spatz_mem_layout.h"
#include "batchnorm_fp16_spatz_params.h"
#include "batchnorm_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int init_input_params(void *params, const float16 *X, const float16 *scale, const float16 *B, const float16 *input_mean, const float16* input_var, const float16 epsilon)
{
    volatile batchnorm_fp16_spatz_params_t *batchnorm_params;
    uint32_t shard;
    uint32_t left;
    uint32_t c_start;
    uint32_t c_end;
    uint32_t c_len;
    uint32_t hw_len;
    uint32_t local_idx;

    batchnorm_params = (volatile batchnorm_fp16_spatz_params_t *) params;

    shard = INPUT0_DIM1 / NUM_HARTS;
    left  = INPUT0_DIM1 % NUM_HARTS;

    c_start = HID * shard + (HID < left ? HID : left);
    c_end   = c_start + shard + (HID < left ? 1 : 0);
    c_len   = c_end - c_start;
    hw_len  = CHANNEL_LEN;

    local_idx = 0;
    for (int c = c_start; c < c_end; c++) {
        uint32_t c_base;

        c_base = c * hw_len;

        for (int i = 0; i < hw_len; i++) {
            uint32_t global_idx;
            uint32_t offset;

            global_idx = c_base + i;
            offset = local_idx * sizeof(float16);

            mmio_fp16(SHARD_X_BASE + offset) = X[global_idx];
            mmio_fp16(SHARD_Y_BASE + offset) = 0;

            local_idx++;
        }
    }

    for (int i = 0; i < INPUT0_DIM1; i++) {
        uint32_t offset = i * sizeof(float16);

        mmio_fp16(MEAN_BASE + offset) = input_mean[i];
        mmio_fp16(VAR_BASE + offset) = input_var[i];
        mmio_fp16(GAMMA_BASE + offset) = scale[i];
        mmio_fp16(BETA_BASE + offset) = B[i];
    }

    mmio_fp16(EPS_BASE) = epsilon;

    batchnorm_params->shard_X = SHARD_X_BASE;
    batchnorm_params->shard_Y = SHARD_Y_BASE;
    batchnorm_params->gamma = GAMMA_BASE;
    batchnorm_params->beta  = BETA_BASE;
    batchnorm_params->mean  = MEAN_BASE;
    batchnorm_params->var   = VAR_BASE;
    batchnorm_params->eps   = EPS_BASE;
    batchnorm_params->c_start = c_start;
    batchnorm_params->c_len   = c_len;
    batchnorm_params->hw_len  = hw_len;

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
    spatz_run_task_with_params(BATCHNORM_FP16_SPATZ_TASK, BATCHNORM_FP16_SPATZ_PARAMS_BASE);

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

static int store_result(void *params, float16 *Y)
{
    volatile batchnorm_fp16_spatz_params_t *batchnorm_params;
    uint32_t shard_Y_base;
    uint32_t local_idx;

    batchnorm_params = (volatile batchnorm_fp16_spatz_params_t *) params;
    shard_Y_base = batchnorm_params->shard_Y;
    local_idx = 0;

    for (int c = 0; c < batchnorm_params->c_len; c++) {
        uint32_t c_base;
        uint32_t c_idx;

        c_idx = batchnorm_params->c_start + c;
        c_base = c_idx * batchnorm_params->hw_len;

        for (int i = 0; i < batchnorm_params->hw_len; i++) {
            uint32_t global_idx;
            uint32_t offset;

            global_idx = c_base + i;
            offset = local_idx * sizeof(float16);

            Y[global_idx] = mmio_fp16(shard_Y_base + offset);

            local_idx++;
        }
    }

    return 0;
}

void MAGIA_batchnorm_fp16_spatz(const float16 *X, const float16 *scale, const float16 *B, const float16 *input_mean, const float16* input_var, const float16 epsilon, float16 *Y)
{
    int ret;
    volatile batchnorm_fp16_spatz_params_t *params;

    params = (volatile batchnorm_fp16_spatz_params_t *) BATCHNORM_FP16_SPATZ_PARAMS_BASE;

    ret = init_input_params(params, X, scale, B, input_mean, input_var, epsilon);
    if (ret != 0) {
        printf("[CV32 (%d)] Params initialization failed with error: %d\n", HID, ret);
        return;
    }

    ret = offload_spatz_task();
    if (ret != 0) {
        printf("[CV32 (%d)] Spatz task offloading failed with error: %d\n", HID, ret);
        return;
    }

    ret = store_result(params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] Result write back failed with error: %d\n", HID, ret);
    }
}
