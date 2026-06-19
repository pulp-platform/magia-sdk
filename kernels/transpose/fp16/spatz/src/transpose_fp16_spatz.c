#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "transpose_fp16_spatz.h"
#include "transpose_fp16_spatz_params.h"
#include "transpose_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int alloc_l1(void **params, uint32_t in_shape[4], uint32_t out_shape[4], uint32_t iterations, uint32_t *out_shard_in_elems, uint32_t *out_shard_out_elems)
{
    volatile transpose_fp16_spatz_params_t *trans_params;
    uintptr_t shard_input;
    uintptr_t shard_output;

    uint32_t shard_in_elems;
    uint32_t shard_out_elems;
    uint32_t iter_start;
    uint32_t iter_end;
    uint32_t iter_len;
    uint32_t shard;
    uint32_t left;

    shard_in_elems  = in_shape[1] * in_shape[2] * in_shape[3];
    shard_out_elems = out_shape[1] * out_shape[2] * out_shape[3];

    shard = iterations / NUM_HARTS;
    left  = iterations % NUM_HARTS;

    iter_start = HID * shard + (HID < left ? HID : left);
    iter_end   = iter_start + shard + (HID < left ? 1 : 0);
    iter_len   = iter_end - iter_start;

    l1_alloc_init();

    trans_params = l1_alloc(sizeof(transpose_fp16_spatz_params_t));
    if (!trans_params)
        return ENOMEM;

    shard_input = l1_alloc(iter_len * shard_in_elems * sizeof(float16));
    if (!shard_input)
        return ENOMEM;

    shard_output = l1_alloc(iter_len * shard_out_elems * sizeof(float16));
    if (!shard_output)
        return ENOMEM;

    trans_params->shard_input    = shard_input;
    trans_params->shard_output   = shard_output;
    trans_params->iteration_start = iter_start;
    trans_params->iteration_len   = iter_len;

    trans_params->out_shape[0]   = out_shape[0];
    trans_params->out_shape[1]   = out_shape[1];
    trans_params->out_shape[2]   = out_shape[2];
    trans_params->out_shape[3]   = out_shape[3];

    trans_params->in_strides[0]  = in_shape[1] * in_shape[2] * in_shape[3];
    trans_params->in_strides[1]  = in_shape[2] * in_shape[3];
    trans_params->in_strides[2]  = in_shape[3];
    trans_params->in_strides[3]  = 1;

    *params = (void *) trans_params;
    *out_shard_in_elems = shard_in_elems;
    *out_shard_out_elems = shard_out_elems;

    return 0;
}

static int init_input_params(void *params, const float16 *input, const uint32_t *perm, uint32_t shard_in_elems, uint32_t shard_out_elems)
{
    volatile transpose_fp16_spatz_params_t *trans_params;
    uintptr_t shard_input_base;
    uintptr_t shard_output_base;
    uint32_t total_in_elems;
    uint32_t total_out_elems;

    trans_params = (volatile transpose_fp16_spatz_params_t *) params;
    shard_input_base  = trans_params->shard_input;
    shard_output_base = trans_params->shard_output;

    total_in_elems  = trans_params->iteration_len * shard_in_elems;
    total_out_elems = trans_params->iteration_len * shard_out_elems;

    uint32_t l1_in_idx = 0;
    uint32_t global_idx_in = trans_params->iteration_start * shard_in_elems;
    for (uint32_t i = 0; i < total_in_elems; i++) {
        uint32_t offset = l1_in_idx * sizeof(float16);
        mmio_fp16(shard_input_base + offset) = input[global_idx_in + i];
        l1_in_idx++;
    }

    uint32_t l1_out_idx = 0;
    for (uint32_t i = 0; i < total_out_elems; i++) {
        uint32_t offset = l1_out_idx * sizeof(float16);
        mmio_fp16(shard_output_base + offset) = 0;
        l1_out_idx++;
    }

    for (int i = 0; i < 4; i++)
        trans_params->perm[i] = perm[i];

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

    spatz_init(SPATZ_BINARY_START);
    spatz_run_task_with_params(TRANSPOSE_FP16_SPATZ_TASK, (uint32_t)params);

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

static int store_result(void *params, float16 *output, uint32_t shard_out_elems)
{
    volatile transpose_fp16_spatz_params_t *trans_params;
    uintptr_t shard_output_base;
    uint32_t total_out_elems;
    uint32_t global_idx_base;
    uint32_t out_idx;

    trans_params = (volatile transpose_fp16_spatz_params_t *) params;
    shard_output_base = trans_params->shard_output;

    total_out_elems = trans_params->iteration_len * shard_out_elems;
    global_idx_base = trans_params->iteration_start * shard_out_elems;
    out_idx = 0;

    for (uint32_t i = 0; i < total_out_elems; i++) {
        uint32_t offset = out_idx * sizeof(float16);
        output[global_idx_base + i] = mmio_fp16(shard_output_base + offset);
        out_idx++;
    }

    return 0;
}

void MAGIA_transpose_fp16_spatz(const float16 *input, float16 *output, uint32_t *perm, uint32_t in_shape[4], uint32_t out_shape[4], uint32_t iterations)
{
    int ret;
    volatile transpose_fp16_spatz_params_t *params;
    uint32_t shard_in_elems;
    uint32_t shard_out_elems;

    ret = alloc_l1((void **)&params, in_shape, out_shape, iterations, &shard_in_elems, &shard_out_elems);
    if (ret != 0) {
        printf("[CV32 (%d)] L1 allocation failed with error: %d\n", HID, ret);
        return;
    }

    ret = init_input_params((void *)params, input, perm, shard_in_elems, shard_out_elems);
    if (ret != 0) {
        printf("[CV32 (%d)] Params initialization failed with error: %d\n", HID, ret);
        return;
    }

    ret = offload_spatz_task((void *)params);
    if (ret != 0) {
        printf("[CV32 (%d)] Spatz task offloading failed with error: %d\n", HID, ret);
        return;
    }

    ret = store_result((void *)params, output, shard_out_elems);
    if (ret != 0) {
        printf("[CV32 (%d)] Result write back failed with error: %d\n", HID, ret);
    }
}
