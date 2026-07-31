#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernels_dma_utils.h"

#include "groupnorm_fp16_spatz.h"
#include "groupnorm_fp16_spatz_params.h"
#include "groupnorm_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "groupnorm_fp16_spatz"

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

    /* Only the groups this tile owns are staged - they are a contiguous run of the input,
     * and the task indexes L1 by the local group. Staging the whole tensor on every tile
     * (which is what indexing L1 by the *global* group used to require) costs 16x the
     * traffic and, with num_groups == 1, stages 128 KB on 15 tiles that do no work. */
    input_len = g_len * elements_per_group;
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

/*
 * The tile's groups are one contiguous run of X, so a single transfer stages them. There
 * is no shard_Y prefill: the task writes every element of every group it owns.
 */
static int init_input_params(void *params, kdma_t *d, const float16 *X, const float16 *scale, const float16 *B, const float16 epsilon)
{
    volatile groupnorm_fp16_spatz_params_t *groupnorm_params;
    uint32_t elements_per_group;

    groupnorm_params = (volatile groupnorm_fp16_spatz_params_t *) params;
    elements_per_group = groupnorm_params->c_per_g * groupnorm_params->hw_len;

    kdma_in(d,
            (uintptr_t)(X + (uintptr_t)groupnorm_params->g_start * elements_per_group),
            groupnorm_params->shard_X,
            groupnorm_params->input_len * sizeof(float16));

    kdma_in(d, (uintptr_t)scale, groupnorm_params->gamma, groupnorm_params->c_out * sizeof(float16));
    kdma_in(d, (uintptr_t)B, groupnorm_params->beta, groupnorm_params->c_out * sizeof(float16));

    mmio_fp16(groupnorm_params->eps) = epsilon;

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
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

/* Mirror of the staging: one contiguous run back out. */
static int store_result(void *params, kdma_t *d, float16 *Y)
{
    volatile groupnorm_fp16_spatz_params_t *groupnorm_params;
    uint32_t elements_per_group;

    groupnorm_params = (volatile groupnorm_fp16_spatz_params_t *) params;
    elements_per_group = groupnorm_params->c_per_g * groupnorm_params->hw_len;

    kdma_out(d,
             (uintptr_t)(Y + (uintptr_t)groupnorm_params->g_start * elements_per_group),
             groupnorm_params->shard_Y,
             groupnorm_params->output_len * sizeof(float16));

    return 0;
}

void MAGIA_groupnorm_fp16_spatz(const float16 *X, float16 *Y, const float16 *scale, const float16 *B, uint32_t input_shape[4], uint32_t num_groups, float16 epsilon)
{
    int ret;
    volatile groupnorm_fp16_spatz_params_t *params;
    kdma_t d;

    ret = alloc_l1(&params, input_shape, num_groups, epsilon);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    /* More tiles than groups: nothing staged, nothing offloaded, nothing written back. */
    if (params->g_len == 0)
        return;

    kdma_open(&d);

    ret = init_input_params(params, &d, X, scale, B, epsilon);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task(params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result(params, &d, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
