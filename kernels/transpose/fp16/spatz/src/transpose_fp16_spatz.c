#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "transpose_fp16_spatz.h"
#include "transpose_fp16_spatz_mem_layout.h"
#include "transpose_fp16_spatz_params.h"
#include "transpose_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int init_input_params(void *params, const float16 *input, const uint32_t *perm, uint32_t iterations)
{
    volatile transpose_fp16_spatz_params_t *trans_params;
    uint32_t shard_in_elems;
    uint32_t shard_out_elems;
    uint32_t iter_start;
    uint32_t iter_end;
    uint32_t iter_len;
    uint32_t shard;
    uint32_t left;

    trans_params = (volatile transpose_fp16_spatz_params_t *) params;

    shard_in_elems = (INPUT0_SIZE / INPUT0_DIM0);
    shard_out_elems = (OUTPUT0_SIZE / OUTPUT0_DIM0);

    shard = iterations / NUM_HARTS;
    left = iterations % NUM_HARTS;

    iter_start = HID * shard + (HID < left ? HID : left);
    iter_end = iter_start + shard + (HID < left ? 1 : 0);
    iter_len = iter_end - iter_start;

    uint32_t l1_in_idx = 0;
    uint32_t global_idx_in = iter_start * shard_in_elems;
    for (int i = 0; i < (iter_len * shard_in_elems); i++) {
        uint32_t offset = l1_in_idx * sizeof(float16);
        mmio_fp16(SHARD_INPUT_BASE + offset) = input[global_idx_in + i];
        l1_in_idx++;
    }

    uint32_t l1_out_idx = 0;
    for (int i = 0; i < (iter_len * shard_out_elems); i++) {
        uint32_t offset = l1_out_idx * sizeof(float16);
        mmio_fp16(SHARD_OUTPUT_BASE + offset) = 0;
        l1_out_idx++;
    }

    trans_params->shard_input     = SHARD_INPUT_BASE;
    trans_params->shard_output    = SHARD_OUTPUT_BASE;
    trans_params->iteration_start = iter_start;
    trans_params->iteration_len   = iter_len;

    trans_params->out_shape[0]    = OUTPUT0_DIM0;
    trans_params->out_shape[1]    = OUTPUT0_DIM1;
    trans_params->out_shape[2]    = OUTPUT0_DIM2;
    trans_params->out_shape[3]    = OUTPUT0_DIM3;

    trans_params->in_strides[0]   = (INPUT0_DIM1 * INPUT0_DIM2 * INPUT0_DIM3);
    trans_params->in_strides[1]   = (INPUT0_DIM2 * INPUT0_DIM3);
    trans_params->in_strides[2]   = (INPUT0_DIM3);
    trans_params->in_strides[3]   = (1);

    for (int i = 0; i < 4; i++)
        trans_params->perm[i] = perm[i];

    return 0;
}

static int offload_spatz_task(void)
{
    eu_controller_t eu_ctrl;
    eu_config_t eu_cfg;
    int ret;

    eu_cfg.hartid = HID;
    eu_ctrl.base = NULL;
    eu_ctrl.cfg = &eu_cfg;
    eu_ctrl.api = &eu_api;

    spatz_init(SPATZ_BINARY_START);
    spatz_run_task_with_params(TRANSPOSE_FP16_SPATZ_TASK, TRANSPOSE_FP16_SPATZ_PARAMS_BASE);

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
    volatile transpose_fp16_spatz_params_t *trans_params;
    uint32_t shard_output_base;
    uint32_t start;
    uint32_t len;

    trans_params = (volatile transpose_fp16_spatz_params_t *) params;
    shard_output_base = trans_params->shard_output;
    start = trans_params->iteration_start;
    len = trans_params->iteration_len;

    uint32_t shard_out_elems = (OUTPUT0_SIZE / OUTPUT0_DIM0);
    uint32_t global_idx_base = start * shard_out_elems;
    uint32_t out_idx = 0;

    for (int i = 0; i < (len * shard_out_elems); i++) {
        uint32_t offset = out_idx * sizeof(float16);
        output[global_idx_base + i] = mmio_fp16(shard_output_base + offset);
        out_idx++;
    }

    return 0;
}

void MAGIA_transpose_fp16_spatz(const float16 *input, float16 *output, uint32_t *perm, uint32_t iterations)
{
    int ret;
    volatile transpose_fp16_spatz_params_t *params;

    params = (volatile transpose_fp16_spatz_params_t *) TRANSPOSE_FP16_SPATZ_PARAMS_BASE;

    ret = init_input_params(params, input, perm, iterations);
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
