#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"

#include "convtranspose_fp16_spatz.h"
#include "convtranspose_fp16_spatz_params.h"
#include "convtranspose_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "convtranspose_fp16_spatz"

static int alloc_l1(void **params, uint32_t input_shape[4], uint32_t output_shape[4], uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t num_groups)
{
    volatile convtranspose_fp16_spatz_params_t *conv_params;
    uintptr_t shard_X;
    uintptr_t shard_W;
    uintptr_t shard_Y;

    size_t n = input_shape[0];
    size_t c_in = input_shape[1];
    size_t c_out = output_shape[1];

    size_t iterations = n * c_out;
    size_t shard = iterations / NUM_HARTS;
    size_t left = iterations % NUM_HARTS;

    size_t iter_start = HID * shard + (HID < left ? HID : left);
    size_t iter_end = iter_start + shard + (HID < left ? 1 : 0);
    size_t iter_len = iter_end - iter_start;

    size_t input_HW_len = input_shape[2] * input_shape[3];
    size_t weight_HW_len = kernel_h * kernel_w;
    size_t output_HW_len = output_shape[2] * output_shape[3];

    l1_alloc_init();

    conv_params = l1_alloc(sizeof(convtranspose_fp16_spatz_params_t));
    if (!conv_params)
        return ENOMEM;

    shard_X = l1_alloc(n * c_in * input_HW_len * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_W = l1_alloc(c_in * (c_out / num_groups) * weight_HW_len * sizeof(float16));
    if (!shard_W)
        return ENOMEM;

    shard_Y = l1_alloc(iter_len * output_HW_len * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    conv_params->shard_X    = shard_X;
    conv_params->shard_W    = shard_W;
    conv_params->shard_Y    = shard_Y;
    conv_params->n_batches  = (uint32_t) n;
    conv_params->c_in       = (uint32_t) c_in;
    conv_params->c_out      = (uint32_t) c_out;
    conv_params->iter_start = (uint32_t) iter_start;
    conv_params->iter_len   = (uint32_t) iter_len;
    conv_params->c_in_g     = (uint32_t) (c_in / num_groups);
    conv_params->c_out_g    = (uint32_t) (c_out / num_groups);
    conv_params->h_in       = input_shape[2];
    conv_params->w_in       = input_shape[3];
    conv_params->h_out      = output_shape[2];
    conv_params->w_out      = output_shape[3];
    conv_params->kernel_h   = kernel_h;
    conv_params->kernel_w   = kernel_w;
    conv_params->stride_h   = stride_h;
    conv_params->stride_w   = stride_w;
    conv_params->pad_h      = pad_h;
    conv_params->pad_w      = pad_w;

    *params = (void *) conv_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X, const float16 *W)
{
    volatile convtranspose_fp16_spatz_params_t *conv_params = (volatile convtranspose_fp16_spatz_params_t *) params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;

    uint32_t total_in = conv_params->n_batches * conv_params->c_in * conv_params->h_in * conv_params->w_in;
    uint32_t total_w  = conv_params->c_in * conv_params->c_out_g * conv_params->kernel_h * conv_params->kernel_w;
    uint32_t total_out = conv_params->iter_len * conv_params->h_out * conv_params->w_out;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* Every tile needs the full input and full weights, both contiguous in L2 -> 1D transfers. */
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) X, (uint32_t) conv_params->shard_X, total_in * sizeof(float16));
    eu_idma_wait_a2o(&eu_ctrl, WFE);
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) W, (uint32_t) conv_params->shard_W, total_w * sizeof(float16));
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    /* ConvTranspose scatters with accumulation (the task reads-modifies-writes dst), so the
       output shard must start zeroed. */
    for (uint32_t i = 0; i < total_out; i++)
        mmio_fp16(conv_params->shard_Y + i * sizeof(float16)) = 0.0f;

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    int ret;

    eu_ctrl_init(&eu_ctrl);
    spatz_run_task_with_params(CONVTRANSPOSE_FP16_SPATZ_TASK, params);

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
    volatile convtranspose_fp16_spatz_params_t *conv_params = (volatile convtranspose_fp16_spatz_params_t *) params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t out_hw_len = conv_params->h_out * conv_params->w_out;
    uint32_t iter_start = conv_params->iter_start;
    uint32_t iter_len = conv_params->iter_len;

    if (iter_len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's output planes [iter_start, iter_start+iter_len) are contiguous in L2. */
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t) (Y + iter_start * out_hw_len), (uint32_t) conv_params->shard_Y, iter_len * out_hw_len * sizeof(float16));
    eu_idma_wait_o2a(&eu_ctrl, WFE);

    return 0;
}

void MAGIA_convtranspose_fp16_spatz(const float16 *X, const float16 *W, float16 *Y, uint32_t input_shape[4], uint32_t output_shape[4], uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t num_groups)
{
    int ret;
    volatile convtranspose_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, input_shape, output_shape, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, num_groups);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params(params, X, W);
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
