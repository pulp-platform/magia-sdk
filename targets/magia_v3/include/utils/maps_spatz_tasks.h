#ifndef MAPS_SPATZ_TASKS_H
#define MAPS_SPATZ_TASKS_H

#if MAPS_HAS_ADD_SPATZ_TASK
#include "add_fp16_spatz_params.h"
#endif
#if MAPS_HAS_RELU_SPATZ_TASK
#include "relu_fp16_spatz_params.h"
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
