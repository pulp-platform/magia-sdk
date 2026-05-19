#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "maxpool2d_fp16_spatz.h"
#include "maxpool2d_fp16_spatz_mem_layout.h"
#include "maxpool2d_fp16_spatz_params.h"
#include "maxpool2d_fp16_spatz_task_bin.h"

#define HID get_hartid()

static int init_input_params(void *params, const float16 *X, uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w)
{
    volatile maxpool2d_fp16_spatz_params_t *maxpool_params;
    uint32_t shard;
    uint32_t left;
    uint32_t c_start;
    uint32_t c_end;
    uint32_t c_len;
    uint32_t in_hw_len;
    uint32_t out_hw_len;
    uint32_t local_idx;

    maxpool_params = (volatile maxpool2d_fp16_spatz_params_t *) params;

    shard = INPUT0_DIM1 / NUM_HARTS;
    left  = INPUT0_DIM1 % NUM_HARTS;

    c_start = HID * shard + (HID < left ? HID : left);
    c_end   = c_start + shard + (HID < left ? 1 : 0);
    c_len   = c_end - c_start;

    in_hw_len  = INPUT_HW_LEN;
    out_hw_len = OUTPUT_HW_LEN;

    local_idx = 0;
    for (int c = c_start; c < c_end; c++) {
        uint32_t c_base = c * in_hw_len;

        for (int i = 0; i < in_hw_len; i++) {
            uint32_t global_idx = c_base + i;
            uint32_t offset = local_idx * sizeof(float16);

            mmio_fp16(SHARD_X_BASE + offset) = X[global_idx];
            local_idx++;
        }
    }

    uint32_t total_out_elements = c_len * out_hw_len;
    for (int i = 0; i < total_out_elements; i++) {
        uint32_t offset = i * sizeof(float16);
        mmio_fp16(SHARD_Y_BASE + offset) = 0;
    }

    maxpool_params->shard_X   = SHARD_X_BASE;
    maxpool_params->shard_Y   = SHARD_Y_BASE;
    maxpool_params->c_start   = c_start;
    maxpool_params->c_len     = c_len;
    maxpool_params->h_in      = INPUT0_DIM2;
    maxpool_params->w_in      = INPUT0_DIM3;
    maxpool_params->h_out     = OUTPUT0_DIM2;
    maxpool_params->w_out     = OUTPUT0_DIM3;
    maxpool_params->kernel_h  = kernel_h;
    maxpool_params->kernel_w  = kernel_w;
    maxpool_params->stride_h  = stride_h;
    maxpool_params->stride_w  = stride_w;
    maxpool_params->pad_h     = pad_h;
    maxpool_params->pad_w     = pad_w;

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
    spatz_run_task_with_params(MAXPOOL2D_FP16_SPATZ_TASK, MAXPOOL2D_FP16_SPATZ_PARAMS_BASE);

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
    volatile maxpool2d_fp16_spatz_params_t *maxpool_params;
    uint32_t shard_Y_base;
    uint32_t out_hw_len;
    uint32_t local_idx;

    maxpool_params = (volatile maxpool2d_fp16_spatz_params_t *) params;
    shard_Y_base = maxpool_params->shard_Y;

    out_hw_len = maxpool_params->h_out * maxpool_params->w_out;
    local_idx = 0;

    for (int c = 0; c < maxpool_params->c_len; c++) {
        uint32_t c_base;
        uint32_t c_idx;

        c_idx = maxpool_params->c_start + c;
        c_base = c_idx * out_hw_len;

        for (int i = 0; i < out_hw_len; i++) {
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


void MAGIA_maxpool2d_fp16_spatz(const float16* X, float16 *Y, uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w)
{
    int ret;
    volatile maxpool2d_fp16_spatz_params_t *params;

    params = (volatile maxpool2d_fp16_spatz_params_t *) MAXPOOL2D_FP16_SPATZ_PARAMS_BASE;

    ret = init_input_params(params, X, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
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
