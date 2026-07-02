#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "slice_fp16_spatz.h"
#include "slice_fp16_spatz_params.h"
#include "slice_fp16_spatz_task_bin.h"

#define HID get_hartid()

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
    uint32_t shard_X;
    uint32_t shard_Y;

    size_t slice_dim;
    size_t inner_dim;
    size_t start_outer;
    size_t len_outer;
    size_t x_size;
    size_t y_size;
    size_t global_offset;
    size_t out_slice_dim;

    slice_params = (volatile slice_fp16_spatz_params_t *) params;

    shard_X = slice_params->shard_X;
    shard_Y = slice_params->shard_Y;
    start_outer = slice_params->start_outer;
    len_outer = slice_params->len_outer;
    slice_dim = slice_params->slice_dim;
    inner_dim = slice_params->inner_dim;
    out_slice_dim = slice_params->out_slice_dim;

    x_size = len_outer * slice_dim * inner_dim;
    y_size = len_outer * out_slice_dim * inner_dim;
    global_offset = start_outer * slice_dim * inner_dim;

    for (uint32_t i = 0; i < x_size; i++) {
        mmio_fp16(shard_X + i * sizeof(float16)) = data[global_offset + i];
    }

    for (uint32_t i = 0; i < y_size; i++) {
        mmio_fp16(shard_Y + i * sizeof(float16)) = 0;
    }

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
    spatz_run_task_with_params(SLICE_FP16_SPATZ_TASK, params);

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

static int store_result(void *params, float16 *sliced)
{
    volatile slice_fp16_spatz_params_t *slice_params;
    uint32_t shard_Y_base;
    uint32_t start_outer;
    uint32_t out_stride;
    uint32_t len_outer;
    uint32_t out_slice_dim;
    uint32_t inner_dim;

    slice_params = (volatile slice_fp16_spatz_params_t *) params;
    shard_Y_base = slice_params->shard_Y;
    start_outer = slice_params->start_outer;
    len_outer = slice_params->len_outer;
    out_slice_dim = slice_params->out_slice_dim;
    inner_dim = slice_params->inner_dim;

    out_stride = out_slice_dim * inner_dim;

    uint32_t l1_elem_idx = 0;

    for (uint32_t o = 0; o < len_outer; o++) {
        uint32_t global_element_offset = (start_outer + o) * out_stride;

        for (uint32_t i = 0; i < out_stride; i++) {
            sliced[global_element_offset + i] = mmio_fp16(shard_Y_base + l1_elem_idx * sizeof(float16));
            l1_elem_idx++;
        }
    }

    return 0;
}

void MAGIA_slice_fp16_spatz(const float16 *data, float16 *sliced, uint32_t outer_dim, uint32_t slice_dim, uint32_t inner_dim, uint32_t start_idx, uint32_t out_slice_dim)
{
    int ret;
    volatile slice_fp16_spatz_params_t *params;

    ret = alloc_l1(&params, outer_dim, slice_dim, inner_dim, start_idx, out_slice_dim);
    if (ret != 0) {
        printf("[CV32 (%d)] L1 allocation failed with error: %d\n", HID, ret);
        return;
    }

    ret = init_input_params(params, data);
    if (ret != 0) {
        printf("[CV32 (%d)] Params initialization failed with error: %d\n", HID, ret);
        return;
    }

    ret = offload_spatz_task(params);
    if (ret != 0) {
        printf("[CV32 (%d)] Spatz task offloading failed with error: %d\n", HID, ret);
        return;
    }

    ret = store_result(params, sliced);
    if (ret != 0) {
        printf("[CV32 (%d)] Result write back failed with error: %d\n", HID, ret);
    }
}
