#include <stdint.h>
#include <errno.h>

#include "eventunit.h"
#include "tile.h"

#include "scatter_fp16_spatz.h"
#include "scatter_fp16_spatz_params.h"
#include "scatter_fp16_spatz_task_bin.h"

#define HID get_hartid()
#define KERNEL_NAME "scatter_fp16_spatz"

static int allocate_l1(void **params, uint32_t outer_size, uint32_t inner_size, uint32_t axis, uint32_t data_axis_dim, uint32_t indices_axis_dim)
{
    volatile scatter_fp16_spatz_params_t * scatter_params;
    uintptr_t shard_data;
    uintptr_t shard_indices;
    uintptr_t shard_updates;
    uintptr_t shard_output;

    uint32_t iter_start;
    uint32_t iter_end;
    uint32_t iter_len;
    uint32_t shard;
    uint32_t left;
    uint32_t elems_data;
    uint32_t elems_indices;

    shard = outer_size / NUM_HARTS;
    left = outer_size % NUM_HARTS;

    iter_start = HID * shard + (HID < left ? HID : left);
    iter_end = iter_start + shard + (HID < left ? 1 : 0);
    iter_len = iter_end - iter_start;

    elems_data = iter_len * data_axis_dim * inner_size;
    elems_indices = iter_len * indices_axis_dim * inner_size;

    l1_alloc_init();

    scatter_params = l1_alloc(sizeof(scatter_fp16_spatz_params_t));
    if (!scatter_params)
        return ENOMEM;

    shard_data = l1_alloc(elems_data * sizeof(float16));
    if (!shard_data)
        return ENOMEM;

    shard_indices = l1_alloc(elems_indices * sizeof(uint32_t));
    if (!shard_indices)
        return ENOMEM;

    shard_updates = l1_alloc(elems_indices * sizeof(float16));
    if (!shard_updates)
        return ENOMEM;

    shard_output = l1_alloc(elems_data * sizeof(float16));
    if (!shard_output)
        return ENOMEM;

    scatter_params->shard_data = shard_data;
    scatter_params->shard_indices = shard_indices;
    scatter_params->shard_updates = shard_updates;
    scatter_params->shard_output = shard_output;

    scatter_params->outer_per_tile = iter_len;
    scatter_params->elems_per_tile = elems_data;
    scatter_params->elems_indices_per_tile = elems_indices;
    scatter_params->outer_start = iter_start;
    scatter_params->inner_size = inner_size;
    scatter_params->data_axis_dim = data_axis_dim;
    scatter_params->indices_axis_dim = indices_axis_dim;
    scatter_params->axis = axis;

    *params = (void *) scatter_params;
    return 0;

}

static int init_input_params(void *params, const float16 *input, const int64_t *indices, const float16 *updates)
{
    volatile scatter_fp16_spatz_params_t *scatter_params;

    uintptr_t shard_data;
    uintptr_t shard_output;
    uintptr_t shard_indices;
    uintptr_t shard_updates;

    uint32_t elems_per_tile;
    uint32_t elems_indices_per_tile;
    uint32_t outer_per_tile;
    uint32_t global_offset_data;
    uint32_t global_offset_indices;
    uint32_t iter_start;
    uint32_t inner_size;
    uint32_t data_axis_dim;
    uint32_t indices_axis_dim;

    scatter_params = (volatile scatter_fp16_spatz_params_t *) params;

    shard_data = scatter_params->shard_data;
    shard_output = scatter_params->shard_output;
    shard_indices = scatter_params->shard_indices;
    shard_updates = scatter_params->shard_updates;

    elems_per_tile = scatter_params->elems_per_tile;
    elems_indices_per_tile = scatter_params->elems_indices_per_tile;
    outer_per_tile = scatter_params->outer_per_tile;
    iter_start = scatter_params->outer_start;
    inner_size = scatter_params->inner_size;
    data_axis_dim = scatter_params->data_axis_dim;
    indices_axis_dim = scatter_params->indices_axis_dim;

    global_offset_data = iter_start * data_axis_dim * inner_size;
    global_offset_indices = iter_start * indices_axis_dim * inner_size;

    for (uint32_t i = 0; i < elems_indices_per_tile; i++) {
        mmio32(shard_indices + i * sizeof(uint32_t)) = (int32_t)indices[global_offset_indices + i];
        mmio_fp16(shard_updates + i * sizeof(float16)) = updates[global_offset_indices + i];
    }

    for (uint32_t i = 0; i < elems_per_tile; i++) {
        mmio_fp16(shard_data + i * sizeof(float16))  = input[global_offset_data + i];
        mmio_fp16(shard_output + i * sizeof(float16)) = 0; // Verrà popolato dalla copia iniziale nel task
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

    spatz_run_task_with_params(SCATTER_FP16_SPATZ_TASK, params);

    ret = eu_spatz_wait(&eu_ctrl, WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n", HID, KERNEL_NAME, ret);
        goto exit;
    }

    ret = spatz_get_exit_code();

exit:
    return ret;
}

static int store_result(void *params, float16 *output)
{
    volatile scatter_fp16_spatz_params_t *scatter_params;
    uintptr_t shard_output;
    uint32_t elems_per_tile;
    uint32_t global_offset;
    uint32_t iter_start;
    uint32_t inner_size;
    uint32_t data_axis_dim;

    scatter_params = (volatile scatter_fp16_spatz_params_t *) params;
    shard_output = scatter_params->shard_output;
    elems_per_tile = scatter_params->elems_per_tile;
    iter_start = scatter_params->outer_start;
    inner_size = scatter_params->inner_size;
    data_axis_dim = scatter_params->data_axis_dim;

    global_offset = iter_start * data_axis_dim * inner_size;

    for (uint32_t i = 0; i < elems_per_tile; i++) {
        output[global_offset + i] = mmio_fp16(shard_output + i * sizeof(float16));
    }

    return 0;
}

void MAGIA_scatter_fp16_spatz(const float16 *data, const int64_t *indices, const float16 *updates, float16 *output, uint32_t outer_size, uint32_t inner_size, uint32_t axis, uint32_t data_axis_dim, uint32_t indices_axis_dim)
{
    int ret;
    void *params;

    ret = allocate_l1(&params, outer_size, inner_size, axis, data_axis_dim, indices_axis_dim);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] L1 allocation failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = init_input_params(params, data, indices, updates);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Params initialization failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = offload_spatz_task(params);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n", HID, KERNEL_NAME, ret);
        return;
    }

    ret = store_result(params, output);
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Result write back failed with error: %d\n", HID, KERNEL_NAME, ret);
    }
}
