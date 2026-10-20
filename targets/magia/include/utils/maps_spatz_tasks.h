#ifndef MAPS_SPATZ_TASKS_H
#define MAPS_SPATZ_TASKS_H

#if MAPS_HAS_ADD_SPATZ_TASK
#include "add_fp16_spatz_params.h"
#endif
#if MAPS_HAS_MUL_SPATZ_TASK
#include "mul_bcast_fp16_spatz_params.h"
#endif
#if MAPS_HAS_MATMUL_SPATZ_TASK
#include "matmul_rows_fp16_spatz_params.h"
#endif
#if MAPS_HAS_DEPTHWISE_CONV_SPATZ_TASK
#include "depthwise_conv_fp16_spatz_params.h"
#endif
#if MAPS_HAS_MAXPOOL_SPATZ_TASK
#include "maxpool2d_fp16_spatz_params.h"
#endif
#if MAPS_HAS_SIGMOID_SPATZ_TASK
#include "sigmoid_fp16_spatz_params.h"
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
        runtime->spatz_params_bytes < sizeof(matmul_rows_fp16_spatz_params_t) ||
        ((uintptr_t)runtime->spatz_params & 0xfu) != 0u)
        return -1;

    volatile matmul_rows_fp16_spatz_params_t *params =
        (volatile matmul_rows_fp16_spatz_params_t *)runtime->spatz_params;
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
#if MAPS_HAS_REDUCE_SUM_SPATZ_TASK
#include "reducesum_fp16_spatz_params.h"
#endif
#if MAPS_HAS_REDUCE_MAX_SPATZ_TASK
#include "reducemax_fp16_spatz_params.h"
#endif
#if MAPS_HAS_BINARY_BCAST_SPATZ_TASK
#include "binary_bcast_fp16_spatz_params.h"
#endif

static inline int maps_wait_for_spatz(maps_operation_runtime_t *runtime)
{
    if (runtime->background != 0)
        while (!eu_check_events(EU_SPATZ_DONE_MASK) &&
               runtime->background(runtime->background_ctx))
            ;
    if (eu_spatz_wait(MAPS_WAIT_MODE) == 0)
        return -1;
    return (int)spatz_get_exit_code();
}

#if MAPS_HAS_REDUCE_SUM_SPATZ_TASK
static inline int maps_execute_reducesum_spatz_buffer(
    uint32_t input, uint32_t output, uint32_t reduce_dim, uint32_t elements,
    maps_operation_runtime_t *runtime)
{
    if (!runtime || !runtime->spatz_initialized || runtime->reducesum_fp16_task == 0u ||
        runtime->spatz_params_bytes < sizeof(reducesum_fp16_spatz_params_t))
        return -1;
    volatile reducesum_fp16_spatz_params_t *params =
        (volatile reducesum_fp16_spatz_params_t *)runtime->spatz_params;
    params->shard_X = input;
    params->shard_Y = output;
    params->reduce_dim = reduce_dim;
    params->inner_dim = elements;
    params->outer_start = 0u;
    params->outer_len = 1u;
    spatz_run_task_with_params(runtime->reducesum_fp16_task,
                               (uint32_t)runtime->spatz_params);
    return maps_wait_for_spatz(runtime);
}
#endif

#if MAPS_HAS_REDUCE_MAX_SPATZ_TASK
static inline int maps_execute_reducemax_spatz_buffer(
    uint32_t input, uint32_t output, uint32_t reduce_dim, uint32_t elements,
    maps_operation_runtime_t *runtime)
{
    if (!runtime || !runtime->spatz_initialized || runtime->reducemax_fp16_task == 0u ||
        runtime->spatz_params_bytes < sizeof(reducemax_fp16_spatz_params_t))
        return -1;
    volatile reducemax_fp16_spatz_params_t *params =
        (volatile reducemax_fp16_spatz_params_t *)runtime->spatz_params;
    params->shard_X = input;
    params->shard_Y = output;
    params->reduce_dim = reduce_dim;
    params->inner_dim = elements;
    spatz_run_task_with_params(runtime->reducemax_fp16_task,
                               (uint32_t)runtime->spatz_params);
    return maps_wait_for_spatz(runtime);
}
#endif

#if MAPS_HAS_REDUCE_SUM_SPATZ_TASK
static inline int maps_execute_reducesum_spatz(
    const tile_plan_t *plan, const op_desc_t *op, uint32_t slot,
    maps_operation_runtime_t *runtime)
{
    if (!runtime || !runtime->spatz_initialized ||
        runtime->reducesum_fp16_task == 0u || op->num_inputs != 1u ||
        op->num_outputs != 1u || op->inputs[0].elem_type != ELEM_F16 ||
        op->outputs[0].elem_type != ELEM_F16 || op->inputs[0].rank == 0u ||
        op->params[0] >= op->inputs[0].rank ||
        runtime->spatz_params_bytes < sizeof(reducesum_fp16_spatz_params_t) ||
        ((uintptr_t)runtime->spatz_params & 0xfu) != 0u)
        return -1;

    /* Dense reductions are [outer, reduce, inner], regardless of axis.
     * In particular [N,C,H,1] reduced over H has inner_dim == 1. */
    const subslice_desc_t *input = &op->inputs[0];
    const uint32_t axis = op->params[0];
    uint32_t inner_dim = 1u, outer_len = 1u, stride = sizeof(uint16_t);
    for (uint32_t d = input->rank; d-- > 0u;) {
        if (input->shape[d] == 0u)
            return -2;
        if (input->shape[d] > 1u && input->strides_bytes[d] != stride)
            return maps_execute_reduce_f16(plan, op, slot);
        stride *= input->shape[d];
        if (d > axis) inner_dim *= input->shape[d];
        if (d < axis) outer_len *= input->shape[d];
    }
    stride = sizeof(uint16_t);
    for (uint32_t d = op->outputs[0].rank; d-- > 0u;) {
        if (op->outputs[0].shape[d] > 1u &&
            op->outputs[0].strides_bytes[d] != stride)
            return maps_execute_reduce_f16(plan, op, slot);
        stride *= op->outputs[0].shape[d];
    }
    const uint32_t reduce_dim = input->shape[axis];
    if (outer_len * inner_dim != maps_operation_elems(&op->outputs[0]))
        return -2;

    volatile reducesum_fp16_spatz_params_t *params =
        (volatile reducesum_fp16_spatz_params_t *)runtime->spatz_params;
    params->shard_X = local_subslice_addr(plan, &op->inputs[0], slot);
    params->shard_Y = local_subslice_addr(plan, &op->outputs[0], slot);
    params->reduce_dim = reduce_dim;
    params->inner_dim = inner_dim;
    params->outer_start = 0u;
    params->outer_len = outer_len;
    spatz_run_task_with_params(runtime->reducesum_fp16_task,
                               (uint32_t)runtime->spatz_params);
    return maps_wait_for_spatz(runtime);
}
#endif

#if MAPS_HAS_REDUCE_MAX_SPATZ_TASK
static inline int maps_execute_reducemax_spatz(
    const tile_plan_t *plan, const op_desc_t *op, uint32_t slot,
    maps_operation_runtime_t *runtime)
{
    if (!runtime || !runtime->spatz_initialized ||
        runtime->reducemax_fp16_task == 0u || op->num_inputs != 1u ||
        op->num_outputs != 1u || op->inputs[0].elem_type != ELEM_F16 ||
        op->outputs[0].elem_type != ELEM_F16 || op->inputs[0].rank == 0u ||
        op->params[0] != op->inputs[0].rank - 1u ||
        runtime->spatz_params_bytes < sizeof(reducemax_fp16_spatz_params_t) ||
        ((uintptr_t)runtime->spatz_params & 0xfu) != 0u)
        return -1;

    const uint32_t reduce_dim = op->inputs[0].shape[op->inputs[0].rank - 1u];
    const uint32_t input_elements = maps_operation_elems(&op->inputs[0]);
    const uint32_t output_elements = maps_operation_elems(&op->outputs[0]);
    if (reduce_dim == 0u || input_elements / reduce_dim != output_elements)
        return -2;

    volatile reducemax_fp16_spatz_params_t *params =
        (volatile reducemax_fp16_spatz_params_t *)runtime->spatz_params;
    params->shard_X = local_subslice_addr(plan, &op->inputs[0], slot);
    params->shard_Y = local_subslice_addr(plan, &op->outputs[0], slot);
    params->reduce_dim = reduce_dim;
    params->inner_dim = output_elements;
    spatz_run_task_with_params(runtime->reducemax_fp16_task,
                               (uint32_t)runtime->spatz_params);
    return maps_wait_for_spatz(runtime);
}
#endif

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

#if MAPS_HAS_MUL_SPATZ_TASK || MAPS_HAS_BINARY_BCAST_SPATZ_TASK
/* Shared geometry mode IDs match both task parameter contracts. */
#define MAPS_BCAST_ROW 0u
#define MAPS_BCAST_SCALAR 1u
static inline int maps_mul_broadcast_params(
    const subslice_desc_t *output, const subslice_desc_t *broadcast,
    uint32_t *rows, uint32_t *row_len, uint32_t *mode)
{
    const uint32_t output_elements = maps_operation_elems(output);
    const uint32_t broadcast_elements = maps_operation_elems(broadcast);
    if (broadcast_elements == output_elements) {
        *rows = 1u;
        *row_len = output_elements;
        *mode = MAPS_BCAST_ROW;
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
        *mode = MAPS_BCAST_SCALAR;
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
    *mode = MAPS_BCAST_ROW;
    return 0;
}

#endif

#if MAPS_HAS_MUL_SPATZ_TASK
static inline int maps_execute_mul_spatz(const tile_plan_t *plan,
                                         const op_desc_t *op,
                                         uint32_t slot,
                                         maps_operation_runtime_t *runtime)
{
    if (!runtime || !runtime->spatz_initialized ||
        runtime->mul_bcast_fp16_task == 0u || (op->num_inputs != 1u && op->num_inputs != 2u) ||
        op->num_outputs != 1u || op->inputs[0].elem_type != ELEM_F16 ||
        (op->num_inputs == 2u && op->inputs[1].elem_type != ELEM_F16) ||
        op->outputs[0].elem_type != ELEM_F16 ||
        runtime->spatz_params_bytes < sizeof(mul_bcast_fp16_spatz_params_t) ||
        ((uintptr_t)runtime->spatz_params & 0xfu) != 0u)
        return -1;

    const uint32_t elements = maps_operation_elems(&op->outputs[0]);
    if (op->num_inputs == 1u) {
        if (maps_operation_elems(&op->inputs[0]) != elements)
            return -2;
        volatile mul_bcast_fp16_spatz_params_t *params =
            (volatile mul_bcast_fp16_spatz_params_t *)runtime->spatz_params;
        params->shard_A = local_subslice_addr(plan, &op->inputs[0], slot);
        params->shard_B = 0u;
        params->shard_Y = local_subslice_addr(plan, &op->outputs[0], slot);
        params->rows = 1u;
        params->row_len = elements;
        params->mode = MUL_BCAST_CONSTANT;
        params->constant_bits = op->params[0];
        spatz_run_task_with_params(runtime->mul_bcast_fp16_task,
                                  (uint32_t)runtime->spatz_params);
        return maps_wait_for_spatz(runtime);
    }
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

#if MAPS_HAS_BINARY_BCAST_SPATZ_TASK
static inline int maps_execute_binary_bcast_spatz(const tile_plan_t *plan,
    const op_desc_t *op, uint32_t slot, maps_operation_runtime_t *runtime)
{
    if (!runtime || !runtime->spatz_initialized ||
        runtime->binary_bcast_fp16_task == 0u || op->num_inputs != 2u ||
        op->num_outputs != 1u || op->inputs[0].elem_type != ELEM_F16 ||
        op->inputs[1].elem_type != ELEM_F16 || op->outputs[0].elem_type != ELEM_F16 ||
        runtime->spatz_params_bytes < sizeof(binary_bcast_fp16_spatz_params_t))
        return -1;
    const uint32_t elements = maps_operation_elems(&op->outputs[0]);
    const subslice_desc_t *full = &op->inputs[0];
    const subslice_desc_t *broadcast = &op->inputs[1];
    if (maps_operation_elems(full) != elements) {
        full = &op->inputs[1];
        broadcast = &op->inputs[0];
    }
    uint32_t rows, row_len, mode;
    if (maps_operation_elems(full) != elements ||
        maps_mul_broadcast_params(&op->outputs[0], broadcast, &rows, &row_len, &mode) != 0)
        return -2;
    volatile binary_bcast_fp16_spatz_params_t *params =
        (volatile binary_bcast_fp16_spatz_params_t *)runtime->spatz_params;
    params->shard_A = local_subslice_addr(plan, full, slot);
    params->shard_B = local_subslice_addr(plan, broadcast, slot);
    params->shard_Y = local_subslice_addr(plan, &op->outputs[0], slot);
    params->rows = rows; params->row_len = row_len;
    params->mode = mode == MAPS_BCAST_SCALAR ? BINARY_BCAST_SCALAR : BINARY_BCAST_ROW;
    params->operation = op->kind == OP_SUB ? BINARY_BCAST_SUB : BINARY_BCAST_DIV;
    spatz_run_task_with_params(runtime->binary_bcast_fp16_task, (uint32_t)runtime->spatz_params);
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

#if MAPS_HAS_DEPTHWISE_CONV_SPATZ_TASK
#include "utils/maps_depthwise_im2col.h"
static inline int maps_execute_depthwise_conv_spatz(
    const tile_plan_t *plan, const op_desc_t *op, uint32_t slot,
    maps_operation_runtime_t *runtime)
{
    if (!runtime || !runtime->spatz_initialized || !runtime->depthwise_conv_fp16_task ||
        (op->num_inputs != 2u && op->num_inputs != 3u) || op->num_outputs != 1u ||
        runtime->spatz_params_bytes < sizeof(depthwise_conv_fp16_spatz_params_t) ||
        ((uintptr_t)runtime->spatz_params & 15u) ||
        !op->params[0] || !op->params[1] || !op->params[2] || !op->params[3] || !op->params[6])
        return -1;
    for (uint32_t i = 0; i < 2u; ++i)
        if (op->inputs[i].rank != 4u || op->inputs[i].elem_type != ELEM_F16 || op->inputs[i].elem_bytes != 2u)
            return -2;
    if (op->outputs[0].rank != 4u || op->outputs[0].elem_type != ELEM_F16 || op->outputs[0].elem_bytes != 2u ||
        op->outputs[0].shape[0] > op->inputs[0].shape[0] || op->outputs[0].shape[1] > op->inputs[1].shape[0] || op->inputs[1].shape[1] != 1u)
        return -2;
    if (op->num_inputs == 3u && (op->inputs[2].rank != 1u || op->inputs[2].elem_type != ELEM_F16 || op->inputs[2].shape[0] < op->outputs[0].shape[1]))
        return -2;
    uint32_t input_c = maps_operation_global_offset(plan, &op->inputs[0], 1u);
    uint32_t weight_c = maps_operation_global_offset(plan, &op->inputs[1], 0u);
    for (uint32_t c = 0; c < op->outputs[0].shape[1]; ++c) {
        uint32_t global_c = (weight_c + c) / op->params[6];
        if (global_c < input_c || global_c - input_c >= op->inputs[0].shape[1])
            return -3;
    }
    volatile depthwise_conv_fp16_spatz_params_t *p =
        (volatile depthwise_conv_fp16_spatz_params_t *)runtime->spatz_params;
    p->input = local_subslice_addr(plan, &op->inputs[0], slot);
    p->weight = local_subslice_addr(plan, &op->inputs[1], slot);
    p->bias = op->num_inputs == 3u ? local_subslice_addr(plan, &op->inputs[2], slot) : 0u;
    p->output = local_subslice_addr(plan, &op->outputs[0], slot);
    for (uint32_t axis = 0; axis < 4u; ++axis) {
        p->input_shape[axis] = op->inputs[0].shape[axis];
        p->weight_shape[axis] = op->inputs[1].shape[axis];
        p->output_shape[axis] = op->outputs[0].shape[axis];
        p->input_strides[axis] = op->inputs[0].strides_bytes[axis];
        p->weight_strides[axis] = op->inputs[1].strides_bytes[axis];
        p->output_strides[axis] = op->outputs[0].strides_bytes[axis];
    }
    p->input_origin_c = input_c;
    p->weight_origin_c = weight_c;
    p->input_origin_h = maps_operation_global_offset(plan, &op->inputs[0], 2u);
    p->input_origin_w = maps_operation_global_offset(plan, &op->inputs[0], 3u);
    p->output_origin_h = maps_operation_global_offset(plan, &op->outputs[0], 2u);
    p->output_origin_w = maps_operation_global_offset(plan, &op->outputs[0], 3u);
    p->stride_h = op->params[0]; p->stride_w = op->params[1];
    p->dilation_h = op->params[2]; p->dilation_w = op->params[3];
    p->pad_top = op->params[4]; p->pad_left = op->params[5];
    p->channel_multiplier = op->params[6];
    p->bias_stride = op->num_inputs == 3u ? op->inputs[2].strides_bytes[0] : 0u;
    p->vector_profile = 0u;
    p->vector_direct = 0u;
#if defined(MAPS_DEPTHWISE_VECTOR_DIRECT) && MAPS_DEPTHWISE_VECTOR_DIRECT
    /* Even horizontal stride selects the same half of each aligned word.
     * An even input width guarantees that the other half stays in the row. */
    if (p->input_strides[3] == 2u && p->output_strides[3] == 2u &&
        (p->stride_w & 1u) == 0u && (p->input_shape[3] & 1u) == 0u &&
        ((p->input | p->input_strides[0] | p->input_strides[1] |
          p->input_strides[2]) & 3u) == 0u)
        p->vector_direct = 1u;
#if defined(MAPS_DEPTHWISE_VECTOR_PLAN) && MAPS_DEPTHWISE_VECTOR_PLAN
    /* The planned path supports every positive stride: odd strides use
     * per-lane halfword selection rather than one shift for the whole vector. */
    if (p->input_strides[3] == 2u && p->output_strides[3] == 2u &&
        (p->input_shape[3]&1u) == 0u &&
        ((p->input | p->input_strides[0] | p->input_strides[1] |
          p->input_strides[2])&3u) == 0u &&
        p->weight_shape[2] && p->weight_shape[3] &&
        (uint64_t)p->weight_shape[2]*p->weight_shape[3] <= 49u)
        p->vector_direct = 2u;
#endif
#endif
#if MAPS_DEPTHWISE_VECTOR_AUTO
#if !defined(MAPS_DEPTHWISE_VECTOR_DIRECT) || !MAPS_DEPTHWISE_VECTOR_DIRECT
    if (maps_depthwise_prefer_vector(p)) p->vector_direct=2u;
#endif
#endif
#if defined(MAPS_DEPTHWISE_PROFILE) && MAPS_DEPTHWISE_PROFILE
    if (p->vector_direct == 2u && runtime->spatz_params_bytes >= sizeof(*p)+32u)
        p->vector_profile = ((uintptr_t)p + sizeof(*p) + 15u) & ~(uintptr_t)15u;
#endif
    p->im2col = 0u;
    p->im2col_columns = 0u;
    p->im2col_stride = 0u;
#if MAPS_DEPTHWISE_IM2COL_GEMM
    if (!p->vector_direct &&
        p->input_strides[3] == 2u && p->output_strides[3] == 2u &&
        p->output_shape[3] != 0u && p->weight_shape[2] != 0u && p->weight_shape[3] != 0u) {
        const uint64_t kernel = (uint64_t)p->weight_shape[2] * p->weight_shape[3];
        const uint32_t header = (sizeof(*p) + 15u) & ~15u;
        /* Reserve an even number of columns for each matrix row and one
         * output row. Every vector load/store then starts four-byte aligned. */
        const uint64_t row_bytes = 2u * (kernel + 1u) * ((uint64_t)p->output_shape[3] + 1u);
        if (runtime->spatz_params_bytes > header && row_bytes <= runtime->spatz_params_bytes - header) {
            const uint32_t block_rows = (runtime->spatz_params_bytes - header) / (uint32_t)row_bytes;
            return maps_depthwise_im2col_gemm(runtime, p, block_rows);
        }
    }
#endif
    MAPS_DW_STAMP(direct_start);
    MAPS_DW_COUNT(4, 1u);
    spatz_run_task_with_params(runtime->depthwise_conv_fp16_task, (uint32_t)p);
    const int direct_result = maps_wait_for_spatz(runtime);
    MAPS_DW_TIME(1, direct_start);
    return direct_result;
}
#endif

#if MAPS_HAS_MAXPOOL_SPATZ_TASK
static inline int maps_execute_maxpool_spatz(const tile_plan_t *plan,
    const op_desc_t *op, uint32_t slot, maps_operation_runtime_t *runtime)
{
    if (!runtime || !runtime->spatz_initialized || !runtime->maxpool_fp16_task ||
        op->num_inputs!=1u || op->num_outputs!=1u ||
        runtime->spatz_params_bytes<sizeof(maxpool2d_fp16_spatz_params_t) ||
        ((uintptr_t)runtime->spatz_params&15u)) return -1;
    const subslice_desc_t *x=&op->inputs[0], *y=&op->outputs[0];
    if (x->rank!=4u || y->rank!=4u || x->elem_type!=ELEM_F16 || y->elem_type!=ELEM_F16 ||
        x->shape[0]!=y->shape[0] || x->shape[1]!=y->shape[1] ||
        !op->params[0] || !op->params[1] || !op->params[2] || !op->params[3]) return -2;
    for (uint32_t which=0; which<2; ++which) {
        const subslice_desc_t *s=which ? y : x;
        uint32_t stride=2u;
        for (uint32_t d=4; d-- >0;) {
            if (s->shape[d]>1 && s->strides_bytes[d]!=stride) return -3;
            stride*=s->shape[d];
        }
    }
    volatile maxpool2d_fp16_spatz_params_t *p=(volatile maxpool2d_fp16_spatz_params_t *)runtime->spatz_params;
    p->shard_X=local_subslice_addr(plan,x,slot);
    p->shard_Y=local_subslice_addr(plan,y,slot);
    p->c_start=0; p->c_len=x->shape[0]*x->shape[1];
    p->h_in=x->shape[2]; p->w_in=x->shape[3];
    p->h_out=y->shape[2]; p->w_out=y->shape[3];
    p->kernel_h=op->params[0]; p->kernel_w=op->params[1];
    p->stride_h=op->params[2]; p->stride_w=op->params[3];
    p->pad_h=op->params[4]; p->pad_w=op->params[5];
    spatz_run_task_with_params(runtime->maxpool_fp16_task,(uint32_t)p);
    return maps_wait_for_spatz(runtime);
}
#endif

#if MAPS_HAS_SIGMOID_SPATZ_TASK
static inline int maps_execute_sigmoid_spatz(const tile_plan_t *plan,
                                          const op_desc_t *op,
                                          uint32_t slot,
                                          maps_operation_runtime_t *runtime)
{
    if (!runtime || !runtime->spatz_initialized ||
        runtime->sigmoid_fp16_task == 0u || op->num_inputs != 1u ||
        op->num_outputs != 1u || op->inputs[0].elem_type != ELEM_F16 ||
        op->outputs[0].elem_type != ELEM_F16 ||
        runtime->spatz_params_bytes < sizeof(sigmoid_fp16_spatz_params_t) ||
        ((uintptr_t)runtime->spatz_params & 0xfu) != 0u)
        return -1;

    const uint32_t elements = maps_operation_elems(&op->outputs[0]);
    if (maps_operation_elems(&op->inputs[0]) != elements)
        return -2;

    for (uint32_t which = 0u; which < 2u; ++which) {
        const subslice_desc_t *sub = which ? &op->outputs[0] : &op->inputs[0];
        uint32_t stride = sizeof(uint16_t);
        for (uint32_t dim = sub->rank; dim-- > 0u;) {
            if (sub->shape[dim] > 1u && sub->strides_bytes[dim] != stride)
                return -3;
            stride *= sub->shape[dim];
        }
        if ((local_subslice_addr(plan, sub, slot) & 3u) != 0u)
            return -4;
    }
    volatile sigmoid_fp16_spatz_params_t *params =
        (volatile sigmoid_fp16_spatz_params_t *)runtime->spatz_params;
    params->shard_X = local_subslice_addr(plan, &op->inputs[0], slot);
    params->shard_Y = local_subslice_addr(plan, &op->outputs[0], slot);
    params->start = 0u;
    params->end = elements;
    params->len = elements;
    spatz_run_task_with_params(
        runtime->sigmoid_fp16_task, (uint32_t)runtime->spatz_params);
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
