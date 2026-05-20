#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "layernorm_fp16_spatz.h"
#include "layernorm_fp16_spatz_mem_layout.h"
#include "layernorm_fp16_spatz_params.h"
#include "layernorm_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int init_input_params(void *params, const float16 *X, const float16 *scale, const float16 *B, const float16 epsilon)
{
    volatile layernorm_fp16_spatz_params_t *layernorm_params;
    uint32_t shard;
    uint32_t left;
    uint32_t r_start;
    uint32_t r_end;
    uint32_t r_len;
    uint32_t w_len;
    uint32_t local_idx;

    layernorm_params = (volatile layernorm_fp16_spatz_params_t *) params;

    shard = TOTAL_ROWS / NUM_HARTS;
    left  = TOTAL_ROWS % NUM_HARTS;

    r_start = HID * shard + (HID < left ? HID : left);
    r_end   = r_start + shard + (HID < left ? 1 : 0);
    r_len   = r_end - r_start;
    w_len   = ROW_LEN;

    local_idx = 0;
    for (int r = r_start; r < r_end; r++) {
        uint32_t r_base;

        r_base = r * w_len;

        for (int i = 0; i < w_len; i++) {
            uint32_t global_idx;
            uint32_t offset;

            global_idx = r_base + i;
            offset = local_idx * sizeof(float16);

            mmio_fp16(SHARD_X_BASE + offset) = X[global_idx];
            mmio_fp16(SHARD_Y_BASE + offset) = 0;

            local_idx++;
        }
    }

    for (int i = 0; i < ROW_LEN; i++) {
        uint32_t offset = i * sizeof(float16);

        mmio_fp16(GAMMA_BASE + offset) = scale[i];
        mmio_fp16(BETA_BASE + offset) = B[i];
    }

    mmio_fp16(EPS_BASE) = epsilon;

    layernorm_params->shard_X = SHARD_X_BASE;
    layernorm_params->shard_Y = SHARD_Y_BASE;
    layernorm_params->gamma   = GAMMA_BASE;
    layernorm_params->beta    = BETA_BASE;
    layernorm_params->eps     = EPS_BASE;
    layernorm_params->r_start = r_start;
    layernorm_params->r_len   = r_len;
    layernorm_params->w_len   = w_len;

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
    spatz_run_task_with_params(LAYERNORM_FP16_SPATZ_TASK, LAYERNORM_FP16_SPATZ_PARAMS_BASE);

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
    volatile layernorm_fp16_spatz_params_t *layernorm_params;
    uint32_t shard_Y_base;
    uint32_t local_idx;

    layernorm_params = (volatile layernorm_fp16_spatz_params_t *) params;
    shard_Y_base = layernorm_params->shard_Y;
    local_idx = 0;

    for (int r = 0; r < layernorm_params->r_len; r++) {
        uint32_t r_idx;
        uint32_t r_base;

        r_idx = layernorm_params->r_start + r;
        r_base = r_idx * layernorm_params->w_len;

        for (int i = 0; i < layernorm_params->w_len; i++) {
            uint32_t global_idx;
            uint32_t offset;

            global_idx = r_base + i;
            offset = local_idx * sizeof(float16);

            Y[global_idx] = mmio_fp16(shard_Y_base + offset);

            local_idx++;
        }
    }

    return 0;
}

void MAGIA_layernorm_fp16_spatz(const float16 *X, const float16 *scale, const float16 *B, const float16 epsilon, float16 *Y)
{
    int ret;
    volatile layernorm_fp16_spatz_params_t *params;

    params = (volatile layernorm_fp16_spatz_params_t *) LAYERNORM_FP16_SPATZ_PARAMS_BASE;

    ret = init_input_params(params, X, scale, B, epsilon);
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
