#ifndef MAPS_SPATZ_TASKS_H
#define MAPS_SPATZ_TASKS_H

#if MAPS_HAS_ADD_SPATZ_TASK
#include "add_fp16_spatz_params.h"
#endif
#if MAPS_HAS_MUL_SPATZ_TASK
#include "mul_bcast_fp16_spatz_params.h"
#endif
#if MAPS_HAS_MATMUL_SPATZ_TASK
#include "matmul_fp16_spatz_params.h"
#endif
#if MAPS_HAS_RELU_SPATZ_TASK
#include "relu_fp16_spatz_params.h"
#endif

static inline int maps_wait_for_spatz(maps_operation_runtime_t *runtime);

#if MAPS_HAS_MATMUL_SPATZ_TASK
static inline int maps_execute_matmul_spatz(
    const tile_plan_t *plan, const op_desc_t *op, uint32_t slot,
    maps_operation_runtime_t *runtime)
{
    if (!runtime || !runtime->spatz_initialized ||
        runtime->matmul_fp16_task == 0u ||
        (op->num_inputs != 2u && op->num_inputs != 3u) ||
        op->num_outputs != 1u || op->params[0] == 0u ||
        op->params[1] == 0u || op->params[2] == 0u ||
        runtime->spatz_params_bytes < sizeof(matmul_fp16_spatz_params_t) ||
        ((uintptr_t)runtime->spatz_params & 0xfu) != 0u)
        return -1;

    volatile matmul_fp16_spatz_params_t *params =
        (volatile matmul_fp16_spatz_params_t *)runtime->spatz_params;
    params->shard_A = local_subslice_addr(plan, &op->inputs[0], slot);
    params->shard_B = local_subslice_addr(plan, &op->inputs[1], slot);
    params->shard_Y = local_subslice_addr(plan, &op->outputs[0], slot);
    params->shard_bias = op->num_inputs == 3u
        ? local_subslice_addr(plan, &op->inputs[2], slot) : 0u;
    params->M = op->params[0];
    params->K = op->params[1];
    params->O = op->params[2];
    params->M_total = params->M;
    params->m_start = 0u;
    params->batch_len = op->params[3];
    params->a_batched = op->params[4];
    params->b_batched = op->params[5];
    params->bias_mode = MATMUL_BIAS_NONE;
    params->bias_batched = 0u;
    params->transpose_b = op->params[7];
    if (op->num_inputs == 3u) {
        const uint32_t bias_elements = maps_operation_elems(&op->inputs[2]);
        uint32_t bias_batch_elements = 0u;
        if (op->params[6] != 0u) {
            params->bias_mode = MATMUL_BIAS_ROW;
            bias_batch_elements = params->M;
        } else {
            const subslice_desc_t *bias = &op->inputs[2];
            const uint32_t last = bias->rank == 0u
                ? 1u : bias->shape[bias->rank - 1u];
            const uint32_t penultimate = bias->rank < 2u
                ? 1u : bias->shape[bias->rank - 2u];
            if (last == params->O && penultimate == params->M) {
                params->bias_mode = MATMUL_BIAS_MATRIX;
                bias_batch_elements = params->M * params->O;
            } else if (last == params->O) {
                params->bias_mode = MATMUL_BIAS_COLUMN;
                bias_batch_elements = params->O;
            } else if (last == 1u && penultimate == params->M) {
                params->bias_mode = MATMUL_BIAS_ROW;
                bias_batch_elements = params->M;
            } else if (last == 1u && penultimate == 1u) {
                params->bias_mode = MATMUL_BIAS_SCALAR;
                bias_batch_elements = 1u;
            } else {
                return -2;
            }
        }
        if (bias_elements == params->batch_len * bias_batch_elements &&
            params->batch_len > 1u)
            params->bias_batched = 1u;
        else if (bias_elements != bias_batch_elements)
            return -2;
    }
    spatz_run_task_with_params(
        runtime->matmul_fp16_task, (uint32_t)runtime->spatz_params);
    return maps_wait_for_spatz(runtime);
}
#endif
#if MAPS_HAS_SOFTMAX_EXP_SPATZ_TASK
#include "softmax_exp_fp16_spatz_params.h"
#endif
#if MAPS_HAS_GROUP_REDUCE_SPATZ_TASK
#include "group_reduce_fp16_spatz_params.h"
#endif
#if MAPS_HAS_GROUP_CENTERED_REDUCE_SPATZ_TASK
#include "group_centered_reduce_fp16_spatz_params.h"
#endif
#if MAPS_HAS_GROUP_NORMALIZE_SPATZ_TASK
#include "group_normalize_fp16_spatz_params.h"
#endif

static inline int maps_wait_for_spatz(maps_operation_runtime_t *runtime)
{
    if (eu_spatz_wait(runtime->eu_ctrl, MAPS_WAIT_MODE) == 0)
        return -1;
    return (int)spatz_get_exit_code();
}

#if MAPS_HAS_ADD_SPATZ_TASK
static inline int maps_execute_add_spatz(const tile_plan_t *plan,
                                         const op_desc_t *op,
                                         uint32_t slot,
                                         maps_operation_runtime_t *runtime)
{
    if (!runtime || !runtime->spatz_initialized ||
        runtime->add_fp16_task == 0u || op->num_inputs != 2u ||
        op->num_outputs != 1u || op->inputs[0].elem_type != ELEM_F16 ||
        op->inputs[1].elem_type != ELEM_F16 ||
        op->outputs[0].elem_type != ELEM_F16 ||
        runtime->spatz_params_bytes < sizeof(add_fp16_spatz_params_t) ||
        ((uintptr_t)runtime->spatz_params & 0xfu) != 0u)
        return -1;

    const uint32_t elements = maps_operation_elems(&op->outputs[0]);
    if (maps_operation_elems(&op->inputs[0]) != elements ||
        maps_operation_elems(&op->inputs[1]) != elements)
        return -2;

    volatile add_fp16_spatz_params_t *params =
        (volatile add_fp16_spatz_params_t *)runtime->spatz_params;
    params->shard_A = local_subslice_addr(plan, &op->inputs[0], slot);
    params->shard_B = local_subslice_addr(plan, &op->inputs[1], slot);
    params->shard_C = local_subslice_addr(plan, &op->outputs[0], slot);
    params->start = 0u;
    params->end = elements;
    params->len = elements;
    spatz_run_task_with_params(
        runtime->add_fp16_task, (uint32_t)runtime->spatz_params);
    return maps_wait_for_spatz(runtime);
}
#endif

#if MAPS_HAS_MUL_SPATZ_TASK
static inline int maps_mul_broadcast_params(
    const subslice_desc_t *output, const subslice_desc_t *broadcast,
    uint32_t *rows, uint32_t *row_len, uint32_t *mode)
{
    const uint32_t output_elements = maps_operation_elems(output);
    const uint32_t broadcast_elements = maps_operation_elems(broadcast);
    if (broadcast_elements == output_elements) {
        *rows = 1u;
        *row_len = output_elements;
        *mode = MUL_BCAST_ROW;
        return 0;
    }
    if (output->rank == 0u || broadcast->rank != output->rank)
        return -1;

    const uint32_t last = output->rank - 1u;
    if (broadcast->shape[last] == 1u) {
        for (uint32_t axis = 0u; axis < last; ++axis)
            if (broadcast->shape[axis] != output->shape[axis])
                goto row_broadcast;
        *rows = broadcast_elements;
        *row_len = output->shape[last];
        *mode = MUL_BCAST_SCALAR;
        return *rows * *row_len == output_elements ? 0 : -1;
    }

row_broadcast:
    *row_len = 1u;
    uint32_t suffix = output->rank;
    while (suffix > 0u &&
           broadcast->shape[suffix - 1u] == output->shape[suffix - 1u]) {
        *row_len *= output->shape[suffix - 1u];
        --suffix;
    }
    for (uint32_t axis = 0u; axis < suffix; ++axis)
        if (broadcast->shape[axis] != 1u)
            return -1;
    if (broadcast_elements != *row_len || output_elements % *row_len != 0u)
        return -1;
    *rows = output_elements / *row_len;
    *mode = MUL_BCAST_ROW;
    return 0;
}

static inline int maps_execute_mul_spatz(const tile_plan_t *plan,
                                         const op_desc_t *op,
                                         uint32_t slot,
                                         maps_operation_runtime_t *runtime)
{
    if (!runtime || !runtime->spatz_initialized ||
        runtime->mul_bcast_fp16_task == 0u || op->num_inputs != 2u ||
        op->num_outputs != 1u || op->inputs[0].elem_type != ELEM_F16 ||
        op->inputs[1].elem_type != ELEM_F16 ||
        op->outputs[0].elem_type != ELEM_F16 ||
        runtime->spatz_params_bytes < sizeof(mul_bcast_fp16_spatz_params_t) ||
        ((uintptr_t)runtime->spatz_params & 0xfu) != 0u)
        return -1;

    const uint32_t elements = maps_operation_elems(&op->outputs[0]);
    const subslice_desc_t *full = &op->inputs[0];
    const subslice_desc_t *broadcast = &op->inputs[1];
    if (maps_operation_elems(full) != elements) {
        full = &op->inputs[1];
        broadcast = &op->inputs[0];
    }
    if (maps_operation_elems(full) != elements)
        return -2;

    uint32_t rows;
    uint32_t row_len;
    uint32_t mode;
    if (maps_mul_broadcast_params(
            &op->outputs[0], broadcast, &rows, &row_len, &mode) != 0)
        return -2;

    volatile mul_bcast_fp16_spatz_params_t *params =
        (volatile mul_bcast_fp16_spatz_params_t *)runtime->spatz_params;
    params->shard_A = local_subslice_addr(plan, full, slot);
    params->shard_B = local_subslice_addr(plan, broadcast, slot);
    params->shard_Y = local_subslice_addr(plan, &op->outputs[0], slot);
    params->rows = rows;
    params->row_len = row_len;
    params->mode = mode;
    spatz_run_task_with_params(
        runtime->mul_bcast_fp16_task, (uint32_t)runtime->spatz_params);
    return maps_wait_for_spatz(runtime);
}
#endif

#if MAPS_HAS_SOFTMAX_EXP_SPATZ_TASK
static inline int maps_execute_softmax_exp_spatz(
    const tile_plan_t *plan, const op_desc_t *op, uint32_t slot,
    maps_operation_runtime_t *runtime)
{
    if (!runtime || !runtime->spatz_initialized ||
        runtime->softmax_exp_fp16_task == 0u || op->num_inputs != 1u ||
        op->num_outputs != 1u ||
        runtime->spatz_params_bytes < sizeof(softmax_exp_fp16_spatz_params_t))
        return -1;
    volatile softmax_exp_fp16_spatz_params_t *params =
        (volatile softmax_exp_fp16_spatz_params_t *)runtime->spatz_params;
    params->input = local_subslice_addr(plan, &op->inputs[0], slot);
    params->output = local_subslice_addr(plan, &op->outputs[0], slot);
    params->len = maps_operation_elems(&op->outputs[0]);
    spatz_run_task_with_params(runtime->softmax_exp_fp16_task,
                               (uint32_t)runtime->spatz_params);
    return maps_wait_for_spatz(runtime);
}
#endif

#if MAPS_HAS_GROUP_REDUCE_SPATZ_TASK
static inline int maps_execute_group_reduce_spatz(
    const tile_plan_t *plan, const op_desc_t *op, uint32_t slot,
    maps_operation_runtime_t *runtime)
{
    if (!runtime || !runtime->spatz_initialized ||
        runtime->group_reduce_fp16_task == 0u || op->num_inputs != 1u ||
        op->num_outputs != 1u ||
        runtime->spatz_params_bytes < sizeof(group_reduce_fp16_spatz_params_t))
        return -1;
    volatile group_reduce_fp16_spatz_params_t *params =
        (volatile group_reduce_fp16_spatz_params_t *)runtime->spatz_params;
    params->input = local_subslice_addr(plan, &op->inputs[0], slot);
    params->output = local_subslice_addr(plan, &op->outputs[0], slot);
    params->local_elements = maps_operation_elems(&op->inputs[0]);
    params->local_spatial_elements = params->local_elements /
        op->inputs[0].shape[1];
    const slice_desc_t *input_slice = get_slice(plan, op->inputs[0].slice_id);
    params->channel_offset = input_slice->full_offset[1];
    params->num_groups = op->params[0];
    params->elements_per_group = op->params[1];
    params->channels_per_group = op->params[2];
    spatz_run_task_with_params(runtime->group_reduce_fp16_task,
                               (uint32_t)runtime->spatz_params);
    return maps_wait_for_spatz(runtime);
}
#endif

#if MAPS_HAS_GROUP_CENTERED_REDUCE_SPATZ_TASK
static inline int maps_execute_group_centered_reduce_spatz(
    const tile_plan_t *plan, const op_desc_t *op, uint32_t slot,
    maps_operation_runtime_t *runtime)
{
    if (!runtime || !runtime->spatz_initialized ||
        runtime->group_centered_reduce_fp16_task == 0u ||
        op->num_inputs != 2u || op->num_outputs != 1u ||
        runtime->spatz_params_bytes <
            sizeof(group_centered_reduce_fp16_spatz_params_t))
        return -1;
    volatile group_centered_reduce_fp16_spatz_params_t *params =
        (volatile group_centered_reduce_fp16_spatz_params_t *)runtime->spatz_params;
    params->input = local_subslice_addr(plan, &op->inputs[0], slot);
    params->mean = local_subslice_addr(plan, &op->inputs[1], slot);
    params->output = local_subslice_addr(plan, &op->outputs[0], slot);
    params->local_elements = maps_operation_elems(&op->inputs[0]);
    params->local_spatial_elements = params->local_elements /
        op->inputs[0].shape[1];
    const slice_desc_t *input_slice = get_slice(plan, op->inputs[0].slice_id);
    params->channel_offset = input_slice->full_offset[1];
    params->num_groups = op->params[0];
    params->elements_per_group = op->params[1];
    params->channels_per_group = op->params[2];
    spatz_run_task_with_params(runtime->group_centered_reduce_fp16_task,
                               (uint32_t)runtime->spatz_params);
    return maps_wait_for_spatz(runtime);
}
#endif

#if MAPS_HAS_GROUP_NORMALIZE_SPATZ_TASK
static inline int maps_execute_group_normalize_spatz(
    const tile_plan_t *plan, const op_desc_t *op, uint32_t slot,
    maps_operation_runtime_t *runtime)
{
    if (!runtime || !runtime->spatz_initialized ||
        runtime->group_normalize_fp16_task == 0u || op->num_inputs != 5u ||
        op->num_outputs != 1u ||
        runtime->spatz_params_bytes < sizeof(group_normalize_fp16_spatz_params_t))
        return -1;
    const slice_desc_t *input_slice = get_slice(plan, op->inputs[0].slice_id);
    const slice_desc_t *scale_slice = get_slice(plan, op->inputs[3].slice_id);
    if (!input_slice || !scale_slice || op->inputs[0].rank < 2u)
        return -2;
    volatile group_normalize_fp16_spatz_params_t *params =
        (volatile group_normalize_fp16_spatz_params_t *)runtime->spatz_params;
    params->input = local_subslice_addr(plan, &op->inputs[0], slot);
    params->mean = local_subslice_addr(plan, &op->inputs[1], slot);
    params->variance = local_subslice_addr(plan, &op->inputs[2], slot);
    params->scale = local_subslice_addr(plan, &op->inputs[3], slot);
    params->bias = local_subslice_addr(plan, &op->inputs[4], slot);
    params->output = local_subslice_addr(plan, &op->outputs[0], slot);
    params->local_elements = maps_operation_elems(&op->outputs[0]);
    params->local_spatial_elements = params->local_elements /
        op->outputs[0].shape[1];
    params->channel_offset = input_slice->full_offset[1];
    params->scale_channel_offset = scale_slice->full_offset[0];
    params->num_groups = op->params[0];
    params->channels_per_group = op->params[3];
    union { uint32_t bits; float value; } epsilon = {.bits = op->params[1]};
    params->epsilon = epsilon.value;
    spatz_run_task_with_params(runtime->group_normalize_fp16_task,
                               (uint32_t)runtime->spatz_params);
    return maps_wait_for_spatz(runtime);
}
#endif

#if MAPS_HAS_RELU_SPATZ_TASK
static inline int maps_execute_relu_spatz(const tile_plan_t *plan,
                                          const op_desc_t *op,
                                          uint32_t slot,
                                          maps_operation_runtime_t *runtime)
{
    if (!runtime || !runtime->spatz_initialized ||
        runtime->relu_fp16_task == 0u || op->num_inputs != 1u ||
        op->num_outputs != 1u || op->inputs[0].elem_type != ELEM_F16 ||
        op->outputs[0].elem_type != ELEM_F16 ||
        runtime->spatz_params_bytes < sizeof(relu_fp16_spatz_params_t) ||
        ((uintptr_t)runtime->spatz_params & 0xfu) != 0u)
        return -1;

    const uint32_t elements = maps_operation_elems(&op->outputs[0]);
    if (maps_operation_elems(&op->inputs[0]) != elements)
        return -2;

    volatile relu_fp16_spatz_params_t *params =
        (volatile relu_fp16_spatz_params_t *)runtime->spatz_params;
    params->shard_X = local_subslice_addr(plan, &op->inputs[0], slot);
    params->shard_Y = local_subslice_addr(plan, &op->outputs[0], slot);
    params->start = 0u;
    params->end = elements;
    params->len = elements;
    spatz_run_task_with_params(
        runtime->relu_fp16_task, (uint32_t)runtime->spatz_params);
    return maps_wait_for_spatz(runtime);
}
#endif

#endif
