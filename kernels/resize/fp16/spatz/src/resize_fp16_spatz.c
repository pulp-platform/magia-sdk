#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"

#include "resize_fp16_spatz.h"
#include "resize_fp16_spatz_params.h"
#include "resize_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "resize_fp16_spatz"

static int alloc_l1(void **params, uint32_t batch_size, uint32_t channels, uint32_t in_h, uint32_t in_w, uint32_t out_h, uint32_t out_w)
{
    volatile resize_fp16_spatz_params_t *resize_params;
    uintptr_t shard_X;
    uintptr_t shard_Y;

    size_t it_start;
    size_t it_end;
    size_t it_len;
    size_t elems;
    size_t left;
    size_t total_iterations;

    total_iterations = batch_size * channels;
    elems = total_iterations / NUM_HARTS;
    left = total_iterations % NUM_HARTS;

    it_start = HID * elems + (HID < left ? HID : left);
    it_end = it_start + elems + (HID < left ? 1 : 0);
    it_len = it_end - it_start;

    l1_alloc_init();

    resize_params = l1_alloc(sizeof(resize_fp16_spatz_params_t));
    if (!resize_params)
        return ENOMEM;

    shard_X = l1_alloc(it_len * in_h * in_w * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(it_len * out_h * out_w * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    resize_params->shard_X = shard_X;
    resize_params->shard_Y = shard_Y;
    resize_params->in_h = in_h;
    resize_params->in_w = in_w;
    resize_params->out_h = out_h;
    resize_params->out_w = out_w;
    resize_params->iteration_start = it_start;
    resize_params->iteration_len = it_len;

    *params = (void *) resize_params;

    return 0;
}

static int init_input_params(void *params, const float16 *X)
{
    volatile resize_fp16_spatz_params_t *resize_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t it_start;
    uint32_t it_len;
    uint32_t in_hw;

    resize_params = (volatile resize_fp16_spatz_params_t *) params;
    it_start = resize_params->iteration_start;
    it_len   = resize_params->iteration_len;
    in_hw    = resize_params->in_h * resize_params->in_w;

    if (it_len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's (batch,channel) planes [it_start, it_start+it_len) are contiguous in L2.
       The Spatz task writes every output pixel, so shard_Y is not zeroed here. */
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (X + it_start * in_hw), (uint32_t) resize_params->shard_X, it_len * in_hw * sizeof(float16));
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    int ret;

    eu_ctrl_init(&eu_ctrl);
    spatz_run_task_with_params(RESIZE_FP16_SPATZ_TASK, params);

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
    volatile resize_fp16_spatz_params_t *resize_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t it_start;
    uint32_t it_len;
    uint32_t out_hw;

    resize_params = (volatile resize_fp16_spatz_params_t *) params;
    it_start = resize_params->iteration_start;
    it_len   = resize_params->iteration_len;
    out_hw   = resize_params->out_h * resize_params->out_w;

    if (it_len == 0)
        return 0;

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's output (batch,channel) planes [it_start, it_start+it_len) are contiguous. */
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t) (Y + it_start * out_hw), (uint32_t) resize_params->shard_Y, it_len * out_hw * sizeof(float16));
    eu_idma_wait_o2a(&eu_ctrl, WFE);

    return 0;
}

void MAGIA_resize_fp16_spatz(const float16 *X, float16 *Y, uint32_t batch_size, uint32_t channels, uint32_t in_h, uint32_t in_w, uint32_t out_h, uint32_t out_w)
{
    int ret;
    volatile resize_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, batch_size, channels, in_h, in_w, out_h, out_w);
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
