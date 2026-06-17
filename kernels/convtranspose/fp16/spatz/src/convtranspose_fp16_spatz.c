#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "convtranspose_fp16_spatz.h"
#include "convtranspose_fp16_spatz_mem_layout.h"
#include "convtranspose_fp16_spatz_params.h"
#include "convtranspose_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int alloc_l1(void **params, uint32_t input_shape[4], uint32_t output_shape[4], uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t num_groups)
{
    volatile convtranspose_fp16_spatz_params_t *convtranspose_params;
    uintptr_t shard_X;
    uintptr_t shard_W;
    uintptr_t shard_Y;

    size_t N_in;
    size_t C_in;
    size_t H_in;
    size_t W_in;
    size_t C_out;
    size_t H_out;
    size_t W_out;

    size_t input_HW_len;
    size_t weight_HW_len;
    size_t output_HW_len;
    size_t c_out_per_tile;

    N_in = input_shape[0];
    C_in = input_shape[1];
    H_in = input_shape[2];
    W_in = input_shape[3];
    C_out = output_shape[1];
    H_out = output_shape[2];
    W_out = output_shape[3];

    input_HW_len = H_in * W_in;
    weight_HW_len = kernel_h * kernel_w;
    output_HW_len = H_out * W_out;
    c_out_per_tile = C_out / NUM_HARTS;

    l1_alloc_init();

    convtranspose_params = l1_alloc(sizeof(convtranspose_fp16_spatz_params_t));
    if (!convtranspose_params)
        return ENOMEM;

    shard_X = l1_alloc(N_in * C_in * input_HW_len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_W = l1_alloc(C_in * c_out_per_tile * weight_HW_len * sizeof(float16));
    if (!shard_W)
        return ENOMEM;

    shard_Y = l1_alloc(N_in * c_out_per_tile * output_HW_len * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    convtranspose_params->shard_X = shard_X;
    convtranspose_params->shard_W = shard_W;
    convtranspose_params->shard_Y = shard_Y;

    *params = (void *) convtranspose_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, const float16 *W, uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t num_groups)
{
    volatile convtranspose_fp16_spatz_params_t *convtranspose_params = (volatile convtranspose_fp16_spatz_params_t *) params;
    uintptr_t shard_X;
    uintptr_t shard_W;
    uintptr_t shard_Y;

    convtranspose_params = (volatile convtranspose_fp16_spatz_params_t *) params;
    shard_X = convtranspose_params->shard_X;
    shard_W = convtranspose_params->shard_W;
    shard_Y = convtranspose_params->shard_Y;

    uint32_t c_in_per_grp;
    uint32_t c_out_per_grp;
    uint32_t c_out_per_grp_left;

    uint32_t c_out_start;
    uint32_t c_out_end;
    uint32_t c_out_len;

    uint32_t group_id;
    uint32_t c_in_start;
    uint32_t c_in_end;
    uint32_t c_in_len;

    c_in_per_grp  = INPUT0_DIM1 / num_groups;
    c_out_per_grp = OUTPUT0_DIM1 / num_groups;
    c_out_per_grp_left = OUTPUT0_DIM1 % NUM_HARTS;

    c_out_start = HID * (OUTPUT0_DIM1 / NUM_HARTS) + (HID < c_out_per_grp_left ? HID : c_out_per_grp_left);
    c_out_end   = c_out_start + (OUTPUT0_DIM1 / NUM_HARTS) + (HID < c_out_per_grp_left ? 1 : 0);
    c_out_len   = c_out_end - c_out_start;

    group_id   = c_out_start / c_out_per_grp;
    c_in_start = group_id * c_in_per_grp;
    c_in_end   = c_in_start + c_in_per_grp;
    c_in_len   = c_in_end - c_in_start;

    uint32_t local_x_idx = 0;
    for (uint32_t n = 0; n < INPUT0_DIM0; n++) {
        for (uint32_t c_in = c_in_start; c_in < c_in_end; c_in++) {
            uint32_t global_X_idx_base = (n * INPUT0_DIM1 * INPUT_HW_LEN) + (c_in * INPUT_HW_LEN);

            for (uint32_t i = 0; i < INPUT_HW_LEN; i++) {
                uint32_t offset = local_x_idx * sizeof(float16);
                mmio_fp16(shard_X + offset) = X[global_X_idx_base + i];
                local_x_idx++;
            }
        }
    }

    uint32_t local_w_idx = 0;
    for (uint32_t c_in = c_in_start; c_in < c_in_end; c_in++) {
        for (uint32_t c_out = c_out_start; c_out < c_out_end; c_out++) {
            uint32_t c_out_local = c_out % c_out_per_grp;
            uint32_t global_W_idx_base = (c_in * c_out_per_grp * WEIGTHS_HW_LEN) + (c_out_local * WEIGTHS_HW_LEN);

            for (uint32_t i = 0; i < WEIGTHS_HW_LEN; i++) {
                uint32_t offset = local_w_idx * sizeof(float16);
                mmio_fp16(shard_W + offset) = W[global_W_idx_base + i];
                local_w_idx++;
            }
        }
    }

    for (uint32_t i = 0; i < (INPUT0_DIM0 * c_out_len * OUTPUT_HW_LEN); i++) {
        mmio_fp16(shard_Y + (i * sizeof(float16))) = 0.0f;
    }

    convtranspose_params->n_batches  = INPUT0_DIM0;
    convtranspose_params->c_out_start = c_out_start;
    convtranspose_params->c_out_len   = c_out_len;
    convtranspose_params->c_in_g      = c_in_len;
    convtranspose_params->c_out_g     = c_out_per_grp;
    convtranspose_params->h_in        = INPUT0_DIM2;
    convtranspose_params->w_in        = INPUT0_DIM3;
    convtranspose_params->h_out       = OUTPUT0_DIM2;
    convtranspose_params->w_out       = OUTPUT0_DIM3;
    convtranspose_params->kernel_h    = kernel_h;
    convtranspose_params->kernel_w    = kernel_w;
    convtranspose_params->stride_h    = stride_h;
    convtranspose_params->stride_w    = stride_w;
    convtranspose_params->pad_h       = pad_h;
    convtranspose_params->pad_w       = pad_w;

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
    spatz_run_task_with_params(CONVTRANSPOSE_FP16_SPATZ_TASK, params);

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
    volatile convtranspose_fp16_spatz_params_t *convtranspose_params = (volatile convtranspose_fp16_spatz_params_t *) params;

    uint32_t out_hw_len = convtranspose_params->h_out * convtranspose_params->w_out;
    uint32_t local_idx = 0;

    for (uint32_t n = 0; n < convtranspose_params->n_batches; n++) {
        for (uint32_t c = 0; c < convtranspose_params->c_out_len; c++) {
            uint32_t global_c = convtranspose_params->c_out_start + c;
            uint32_t global_base = (n * OUTPUT0_DIM1 * out_hw_len) + (global_c * out_hw_len);

            for (uint32_t i = 0; i < out_hw_len; i++) {
                uint32_t global_idx = global_base + i;
                uint32_t offset = local_idx * sizeof(float16);

                Y[global_idx] = mmio_fp16(convtranspose_params->shard_Y + offset);
                local_idx++;
            }
        }
    }

    return 0;
}

void MAGIA_convtranspose_fp16_spatz(const float16 *X, const float16 *W, float16 *Y, uint32_t input_shape[4], uint32_t output_shape[4], uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t num_groups)
{
    int ret;
    volatile convtranspose_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, input_shape, output_shape, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, num_groups);
    if (ret != 0) {
        printf("[CV32 (%d)] L1 allocation failed with error: %d\n", HID, ret);
        return;
    }

    ret = init_input_params(params, X, W, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, num_groups);
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
