#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "groupnorm_fp16_spatz.h"
#include "groupnorm_fp16_spatz_mem_layout.h"
#include "groupnorm_fp16_spatz_params.h"
#include "groupnorm_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int init_input_params(void *params, const float16 *X, const float16 *scale, const float16 *B, const float16 epsilon, const uint32_t num_groups)
{
    volatile groupnorm_fp16_spatz_params_t *groupnorm_params;
    uint32_t shard;
    uint32_t left;
    uint32_t g_start;
    uint32_t g_end;
    uint32_t g_len;
    uint32_t c_per_g;
    uint32_t hw_len;
    uint32_t local_idx;

    groupnorm_params = (volatile groupnorm_fp16_spatz_params_t *) params;

    shard = num_groups / NUM_HARTS;
    left  = num_groups % NUM_HARTS;

    g_start = HID * shard + (HID < left ? HID : left);
    g_end   = g_start + shard + (HID < left ? 1 : 0);
    g_len   = g_end - g_start;
    c_per_g = INPUT0_DIM1 / num_groups;
    hw_len  = CHANNEL_LEN;

    local_idx = 0;
    for (int g = g_start; g < g_end; g++) {
        uint32_t c_start_global = g * c_per_g;
        uint32_t c_end_global   = c_start_global + c_per_g;

        for (int c = c_start_global; c < c_end_global; c++) {
            uint32_t c_base = c * hw_len;

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
    }

    for (int i = 0; i < INPUT0_DIM1; i++) {
        uint32_t offset = i * sizeof(float16);

        mmio_fp16(GAMMA_BASE + offset) = scale[i];
        mmio_fp16(BETA_BASE + offset) = B[i];
    }

    mmio_fp16(EPS_BASE) = epsilon;

    groupnorm_params->shard_X = SHARD_X_BASE;
    groupnorm_params->shard_Y = SHARD_Y_BASE;
    groupnorm_params->gamma = GAMMA_BASE;
    groupnorm_params->beta  = BETA_BASE;
    groupnorm_params->eps   = EPS_BASE;
    groupnorm_params->g_start = g_start;
    groupnorm_params->g_len   = g_len;
    groupnorm_params->c_per_g = c_per_g;
    groupnorm_params->hw_len  = hw_len;

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
    spatz_run_task_with_params(GROUPNORM_FP16_SPATZ_TASK, GROUPNORM_FP16_SPATZ_PARAMS_BASE);

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
    volatile groupnorm_fp16_spatz_params_t *groupnorm_params;
    uint32_t shard_Y_base;
    uint32_t local_idx;

    groupnorm_params = (volatile groupnorm_fp16_spatz_params_t *) params;
    shard_Y_base = groupnorm_params->shard_Y;
    local_idx = 0;

    for (int g = 0; g < groupnorm_params->g_len; g++) {
        uint32_t g_idx = groupnorm_params->g_start + g;
        uint32_t c_start_global = g_idx * groupnorm_params->c_per_g;
        uint32_t c_end_global   = c_start_global + groupnorm_params->c_per_g;

        for (int c = c_start_global; c < c_end_global; c++) {
            uint32_t c_base = c * groupnorm_params->hw_len;

            for (int i = 0; i < groupnorm_params->hw_len; i++) {
                uint32_t global_idx;
                uint32_t offset;

                global_idx = c_base + i;
                offset = local_idx * sizeof(float16);

                Y[global_idx] = mmio_fp16(shard_Y_base + offset);

                local_idx++;
            }
        }
    }

    return 0;
}

void MAGIA_groupnorm_fp16_spatz(const float16 *X, const float16 *Y, const float16 *scale, const float16 *B, uint32_t batch_size, uint32_t num_channels, uint32_t spatial, uint32_t num_groups, float16 epsilon)
{
    int ret;
    volatile groupnorm_fp16_spatz_params_t *params;

    params = (volatile groupnorm_fp16_spatz_params_t *) GROUPNORM_FP16_SPATZ_PARAMS_BASE;

    printf("num_groups: %d\n", num_groups);

    ret = init_input_params(params, X, scale, B, epsilon, num_groups);
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
