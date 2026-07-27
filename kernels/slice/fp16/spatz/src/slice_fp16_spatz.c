#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

#include "kernel_idma_utils.h"

#include "slice_fp16_spatz.h"
#include "slice_fp16_spatz_params.h"
#include "slice_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "slice_fp16_spatz"

static int alloc_l1(void **params, uint32_t outer_dim, uint32_t slice_dim, uint32_t inner_dim, uint32_t start_idx, uint32_t out_slice_dim)
{
    volatile slice_fp16_spatz_params_t *slice_params;
    uint32_t shard_X;
    uint32_t shard_Y;

    size_t start_outer;
    size_t end_outer;
    size_t len_outer;
    size_t elems;
    size_t left;

    size_t x_size;
    size_t y_size;

    elems = outer_dim / NUM_HARTS;
    left = outer_dim % NUM_HARTS;

    start_outer = HID * elems + (HID < left ? HID : left);
    end_outer = start_outer + elems + (HID < left ? 1 : 0);
    len_outer = end_outer - start_outer;

    x_size = len_outer * slice_dim * inner_dim;
    y_size = len_outer * out_slice_dim * inner_dim;

    l1_alloc_init();

    slice_params = l1_alloc(sizeof(slice_fp16_spatz_params_t));
    if (!slice_params)
        return ENOMEM;

    shard_X = l1_alloc(x_size * sizeof(float16));
    if (!shard_X)
        return ENOMEM;

    shard_Y = l1_alloc(y_size * sizeof(float16));
    if (!shard_Y)
        return ENOMEM;

    slice_params->shard_X = shard_X;
    slice_params->shard_Y = shard_Y;
    slice_params->slice_dim = slice_dim;
    slice_params->out_slice_dim = out_slice_dim;
    slice_params->inner_dim = inner_dim;
    slice_params->start_outer = start_outer;
    slice_params->len_outer = len_outer;
    slice_params->start_idx = start_idx;

    *params = (void *) slice_params;

    return 0;
}

static int init_input_params(void *params, const float16 *data)
{
    volatile slice_fp16_spatz_params_t *slice_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t start_outer;
    uint32_t len_outer;
    uint32_t slice_dim;
    uint32_t inner_dim;
    uint32_t global_offset;
    uint32_t x_bytes;

    slice_params = (volatile slice_fp16_spatz_params_t *) params;
    start_outer = slice_params->start_outer;
    len_outer = slice_params->len_outer;
    slice_dim = slice_params->slice_dim;
    inner_dim = slice_params->inner_dim;

    if (len_outer == 0)
        return 0;

    global_offset = start_outer * slice_dim * inner_dim;
    x_bytes = len_outer * slice_dim * inner_dim * sizeof(float16);

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* Load this tile's contiguous [len_outer, slice_dim, inner] block; the Spatz
       task extracts the slice and fully writes shard_Y (so no zeroing needed). */
    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t) (data + global_offset), (uint32_t) slice_params->shard_X, x_bytes);
    eu_idma_wait_a2o(&eu_ctrl, WFE);

    return 0;
}

static int offload_spatz_task(void *params)
{
    eu_controller_t eu_ctrl;
    int ret;

    eu_ctrl_init(&eu_ctrl);

    spatz_run_task_with_params(SLICE_FP16_SPATZ_TASK, params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *sliced)
{
    volatile slice_fp16_spatz_params_t *slice_params;
    idma_controller_t idma_ctrl;
    eu_controller_t eu_ctrl;
    uint32_t start_outer;
    uint32_t len_outer;
    uint32_t out_stride;
    uint32_t global_offset;
    uint32_t y_bytes;

    slice_params = (volatile slice_fp16_spatz_params_t *) params;
    start_outer = slice_params->start_outer;
    len_outer = slice_params->len_outer;
    out_stride = slice_params->out_slice_dim * slice_params->inner_dim;

    if (len_outer == 0)
        return 0;

    global_offset = start_outer * out_stride;
    y_bytes = len_outer * out_stride * sizeof(float16);

    idma_ctrl_init(&idma_ctrl);
    eu_ctrl_init(&eu_ctrl);

    /* This tile's output rows [start_outer, start_outer+len_outer) are contiguous in L2. */
    idma_memcpy_1d(&idma_ctrl, 1, (uint32_t) (sliced + global_offset), (uint32_t) slice_params->shard_Y, y_bytes);
    eu_idma_wait_o2a(&eu_ctrl, WFE);

    return 0;
}

void MAGIA_slice_fp16_spatz(const float16 *data, float16 *sliced, uint32_t outer_dim, uint32_t slice_dim, uint32_t inner_dim, uint32_t start_idx, uint32_t out_slice_dim)
{
    int ret;
    volatile slice_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, outer_dim, slice_dim, inner_dim, start_idx, out_slice_dim);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params(params, data);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task(params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result(params, sliced);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
