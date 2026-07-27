#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"

#include "averagepool2d_fp16_spatz.h"
#include "averagepool2d_fp16_spatz_params.h"
#include "averagepool2d_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "averagepool2d_fp16_spatz"

static int alloc_l1(void **params, uint32_t input_shape[4], uint32_t output_shape[4], uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w)
{
    volatile averagepool2d_fp16_spatz_params_t *averagepool_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;

    size_t input_B;
    size_t input_C;
    size_t input_H;
    size_t input_W;
    size_t output_H;
    size_t output_W;
    size_t iterations;
    size_t input_HW_len;
    size_t output_HW_len;

    size_t c_start;
    size_t c_end;
    size_t c_len;
    size_t elems;
    size_t left;

    input_B = input_shape[0];
    input_C = input_shape[1];
    iterations = input_B * input_C;

    input_H = input_shape[2];
    input_W = input_shape[3];
    output_H = output_shape[2];
    output_W = output_shape[3];

    elems = iterations / NUM_HARTS;
    left = iterations % NUM_HARTS;

    c_start = HID * elems + (HID < left ? HID : left);
    c_end = c_start + elems + (HID < left ? 1 : 0);
    c_len = c_end - c_start;

    input_HW_len = input_H * input_W;
    output_HW_len = output_H * output_W;

    l1_alloc_init();

    averagepool_params = l1_alloc(sizeof(averagepool2d_fp16_spatz_params_t));
    if (!averagepool_params)
        return ENOMEM;

    shard_X = l1_alloc(c_len * input_HW_len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(c_len * output_HW_len * sizeof(float16));
    if(!shard_Y)
        return ENOMEM;

    averagepool_params->shard_X = shard_X;
    averagepool_params->shard_Y = shard_Y;
    averagepool_params->h_out = output_H;
    averagepool_params->w_out = output_W;
    averagepool_params->h_in = input_H;
    averagepool_params->w_in = input_W;
    averagepool_params->c_start = c_start;
    averagepool_params->c_len = c_len;
    averagepool_params->kernel_h = kernel_h;
    averagepool_params->kernel_w = kernel_w;
    averagepool_params->stride_h = stride_h;
    averagepool_params->stride_w = stride_w;
    averagepool_params->pad_h = pad_h;
    averagepool_params->pad_w = pad_w;

    *params = (void *) averagepool_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile averagepool2d_fp16_spatz_params_t *averagepool_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t c_start;
    uint32_t c_len;
    uint32_t in_hw_len;

    averagepool_params = (volatile averagepool2d_fp16_spatz_params_t *) params;
    c_start = averagepool_params->c_start;
    c_len = averagepool_params->c_len;
    in_hw_len = averagepool_params->h_in * averagepool_params->w_in;

    if (c_len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's channels [c_start, c_end) are contiguous in L2 (NCHW). The Spatz
       task writes every output element, so shard_Y is not zeroed here. */
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (X + c_start * in_hw_len), (uint32_t) averagepool_params->shard_X, c_len * in_hw_len * sizeof(float16));
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    int ret;

    eu_ctrl_init(&eu_ctrl);

    spatz_run_task_with_params(AVERAGEPOOL2D_FP16_SPATZ_TASK, params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *Y)
{
    volatile averagepool2d_fp16_spatz_params_t *averagepool_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t c_start;
    uint32_t c_len;
    uint32_t out_hw_len;

    averagepool_params = (volatile averagepool2d_fp16_spatz_params_t *) params;
    c_start = averagepool_params->c_start;
    c_len = averagepool_params->c_len;
    out_hw_len = averagepool_params->h_out * averagepool_params->w_out;

    if (c_len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's output channels [c_start, c_end) are contiguous in L2 (NCHW). */
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t) (Y + c_start * out_hw_len), (uint32_t) averagepool_params->shard_Y, c_len * out_hw_len * sizeof(float16));
    eu_idma_wait_o2a(&eu_ctrl, WFE);

    return 0;
}


void MAGIA_averagepool2d_fp16_spatz(const float16* X, float16 *Y, uint32_t input_shape[4], uint32_t output_shape[4], uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w)
{
    int ret;
    volatile averagepool2d_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, input_shape, output_shape, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params(params, X);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task(params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result(params, Y);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
