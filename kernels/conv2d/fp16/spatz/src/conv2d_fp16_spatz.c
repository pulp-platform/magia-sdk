#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "conv2d_fp16_spatz.h"
#include "conv2d_fp16_spatz_mem_layout.h"
#include "conv2d_fp16_spatz_params.h"
#include "conv2d_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int init_input_params(void *params, const float16 *X, const float16 *W, uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t group)
{
    volatile conv2d_fp16_spatz_params_t *conv_params;
    uint32_t shard;
    uint32_t left;
    uint32_t cout_start;
    uint32_t cout_end;
    uint32_t cout_len;
    uint32_t local_idx;

    conv_params = (volatile conv2d_fp16_spatz_params_t *) params;

    shard = OUTPUT0_DIM1 / NUM_HARTS;
    left  = OUTPUT0_DIM1 % NUM_HARTS;

    cout_start = HID * shard + (HID < left ? HID : left);
    cout_end   = cout_start + shard + (HID < left ? 1 : 0);
    cout_len   = cout_end - cout_start;

    uint32_t total_in_elements = FULL_INPUT_LEN;
    for (int i = 0; i < total_in_elements; i++) {
        uint32_t offset = i * sizeof(float16);
        mmio_fp16(SHARD_X_BASE + offset) = X[i];
    }

    uint32_t weight_h_w_len = kernel_h * kernel_w;
    uint32_t weight_cin_len = INPUT0_DIM1;  /* C_in / group */
    uint32_t weight_cout_stride = weight_cin_len * weight_h_w_len;

    local_idx = 0;
    for (int cout = cout_start; cout < cout_end; cout++) {
        uint32_t w_global_base = cout * weight_cout_stride;
        for (int i = 0; i < weight_cout_stride; i++) {
            uint32_t offset = local_idx * sizeof(float16);
            mmio_fp16(SHARD_W_BASE + offset) = W[w_global_base + i];
            local_idx++;
        }
    }

    uint32_t total_out_elements = cout_len * OUTPUT_HW_LEN;
    for (int i = 0; i < total_out_elements; i++) {
        uint32_t offset = i * sizeof(float16);
        mmio_fp16(SHARD_Y_BASE + offset) = 0;
    }

    conv_params->shard_X     = SHARD_X_BASE;
    conv_params->shard_W     = SHARD_W_BASE;
    conv_params->shard_Y     = SHARD_Y_BASE;
    conv_params->c_out_start = cout_start;
    conv_params->c_out_len   = cout_len;
    conv_params->c_in_g      = INPUT0_DIM1 / group;
    conv_params->c_out_g     = OUTPUT0_DIM1 / group;
    conv_params->h_in        = INPUT0_DIM2;
    conv_params->w_in        = INPUT0_DIM3;
    conv_params->h_out       = OUTPUT0_DIM2;
    conv_params->w_out       = OUTPUT0_DIM3;
    conv_params->kernel_h    = kernel_h;
    conv_params->kernel_w    = kernel_w;
    conv_params->stride_h    = stride_h;
    conv_params->stride_w    = stride_w;
    conv_params->pad_h       = pad_h;
    conv_params->pad_w       = pad_w;

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
    spatz_run_task_with_params(CONV2D_FP16_SPATZ_TASK, CONV2D_FP16_SPATZ_PARAMS_BASE);

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
    volatile conv2d_fp16_spatz_params_t *conv_params;
    uint32_t shard_Y_base;
    uint32_t out_hw_len;
    uint32_t local_idx;

    conv_params = (volatile conv2d_fp16_spatz_params_t *) params;
    shard_Y_base = conv_params->shard_Y;

    out_hw_len = conv_params->h_out * conv_params->w_out;
    local_idx = 0;

    for (int c = 0; c < conv_params->c_out_len; c++) {
        uint32_t cout_global_idx = conv_params->c_out_start + c;
        uint32_t y_global_base = cout_global_idx * out_hw_len;

        for (int i = 0; i < out_hw_len; i++) {
            uint32_t offset = local_idx * sizeof(float16);
            Y[y_global_base + i] = mmio_fp16(shard_Y_base + offset);
            local_idx++;
        }
    }

    return 0;
}

void MAGIA_conv2d_fp16_spatz(const float16* X, const float16 *W, float16 *Y, uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t group)
{
    int ret;
    volatile conv2d_fp16_spatz_params_t *params;

    params = (volatile conv2d_fp16_spatz_params_t *) CONV2D_FP16_SPATZ_PARAMS_BASE;

    ret = init_input_params(params, X, W, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, group);
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
