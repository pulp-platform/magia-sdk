#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "groupnorm_fp16_spatz.h"
#include "groupnorm_fp16_spatz_params.h"
#include "groupnorm_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int alloc_l1(void **params, uint32_t input_shape[4], uint32_t num_groups, float16 epsilon)
{
    volatile groupnorm_fp16_spatz_params_t *groupnorm_params;
    uintptr_t shard_X;
    uintptr_t shard_W;
    uintptr_t shard_Y;
    uintptr_t gamma;
    uintptr_t beta;
    uintptr_t eps;

    size_t n;
    size_t c_in;
    size_t h_in;
    size_t w_in;
    size_t iterations;
    size_t shard;
    size_t left;
    size_t g_start;
    size_t g_end;
    size_t g_len;
    size_t c_per_g;
    size_t hw_len;
    size_t elements_per_group;
    size_t input_len;
    size_t output_len;

    n = input_shape[0];
    c_in = input_shape[1];
    h_in = input_shape[2];
    w_in = input_shape[3];

    iterations = n * num_groups;
    shard = iterations / NUM_HARTS;
    left = iterations % NUM_HARTS;

    g_start = HID * shard + (HID < left ? HID : left);
    g_end = g_start + shard + (HID < left ? 1 : 0);
    g_len = g_end - g_start;

    c_per_g = c_in / num_groups;
    hw_len = h_in * w_in;
    elements_per_group = c_per_g * hw_len;

    input_len = n * c_in * hw_len;
    output_len = g_len * elements_per_group;

    l1_alloc_init();

    groupnorm_params = l1_alloc(sizeof(groupnorm_fp16_spatz_params_t));
    if (!groupnorm_params)
        return ENOMEM;

    shard_X = l1_alloc(input_len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(output_len * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    gamma = l1_alloc(c_in * sizeof(float16));
    if (!gamma)
        return ENOMEM;

    beta = l1_alloc(c_in * sizeof(float16));
    if (!beta)
        return ENOMEM;

    eps = l1_alloc(sizeof(float16));
    if (!eps)
        return ENOMEM;

    groupnorm_params->shard_X   = shard_X;
    groupnorm_params->shard_Y   = shard_Y;
    groupnorm_params->gamma     = gamma;
    groupnorm_params->beta      = beta;
    groupnorm_params->eps       = eps;
    groupnorm_params->g_start   = (uint32_t) g_start;
    groupnorm_params->g_len     = (uint32_t) g_len;
    groupnorm_params->c_per_g   = (uint32_t) c_per_g;
    groupnorm_params->hw_len    = (uint32_t) hw_len;
    groupnorm_params->num_groups = (uint32_t) num_groups;
    groupnorm_params->c_out     = (uint32_t) c_in;
    groupnorm_params->input_len = (uint32_t) input_len;
    groupnorm_params->output_len = (uint32_t) output_len;

    *params = (void *) groupnorm_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, const float16 *scale, const float16 *B, const float16 epsilon)
{
    volatile groupnorm_fp16_spatz_params_t *groupnorm_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;
    uintptr_t gamma;
    uintptr_t beta;
    uintptr_t eps;

    groupnorm_params = (volatile groupnorm_fp16_spatz_params_t *) params;
    shard_X = groupnorm_params->shard_X;
    shard_Y = groupnorm_params->shard_Y;
    gamma = groupnorm_params->gamma;
    beta = groupnorm_params->beta;
    eps = groupnorm_params->eps;

    for (uint32_t i = 0; i < groupnorm_params->input_len; i++)
        mmio_fp16(shard_X + i * sizeof(float16)) = X[i];

    for (uint32_t i = 0; i < groupnorm_params->output_len; i++)
        mmio_fp16(shard_Y + i * sizeof(float16)) = 0;

    for (uint32_t i = 0; i < groupnorm_params->c_out; i++) {
        uint32_t offset = i * sizeof(float16);
        mmio_fp16(gamma + offset) = scale[i];
        mmio_fp16(beta + offset) = B[i];
    }

    mmio_fp16(eps) = epsilon;

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

    spatz_run_task_with_params(GROUPNORM_FP16_SPATZ_TASK, (uint32_t)params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] Wait on Spatz task completion failed with error: %d\n", HID, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *Y)
{
    volatile groupnorm_fp16_spatz_params_t *groupnorm_params;
    uintptr_t shard_Y_base;
    uint32_t elements_per_group;
    uint32_t local_idx;

    groupnorm_params = (volatile groupnorm_fp16_spatz_params_t *) params;
    shard_Y_base = groupnorm_params->shard_Y;
    elements_per_group = groupnorm_params->c_per_g * groupnorm_params->hw_len;
    local_idx = 0;

    for (uint32_t g = 0; g < groupnorm_params->g_len; g++) {
        uint32_t global_g = groupnorm_params->g_start + g;
        uint32_t global_group_offset = global_g * elements_per_group;

        for (uint32_t i = 0; i < elements_per_group; i++) {
            uint32_t offset = local_idx * sizeof(float16);
            Y[global_group_offset + i] = mmio_fp16(shard_Y_base + (uintptr_t)offset);
            local_idx++;
        }
    }

    return 0;
}

void MAGIA_groupnorm_fp16_spatz(const float16 *X, const float16 *Y, const float16 *scale, const float16 *B, uint32_t input_shape[4], uint32_t num_groups, float16 epsilon)
{
    int ret;
    volatile groupnorm_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, input_shape, num_groups, epsilon);
    if (ret != 0) {
        printf("[CV32 (%d)] L1 allocation failed with error: %d\n", HID, ret);
        return;
    }

    ret = init_input_params(params, X, scale, B, epsilon);
    if (ret != 0) {
        printf("[CV32 (%d)] Params initialization failed with error: %d\n", HID, ret);
        return;
    }

    ret = offload_spatz_task(params);
    if (ret != 0) {
        printf("[CV32 (%d)] Spatz task offloading failed with error: %d\n", HID, ret);
        return;
    }

    ret = store_result(params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] Result write back failed with error: %d\n", HID, ret);
    }
}
