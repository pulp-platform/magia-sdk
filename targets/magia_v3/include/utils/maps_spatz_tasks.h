#ifndef MAPS_SPATZ_TASKS_H
#define MAPS_SPATZ_TASKS_H

#if MAPS_HAS_ADD_SPATZ_TASK
#include "add_fp16_spatz_params.h"
#endif
#if MAPS_HAS_RELU_SPATZ_TASK
#include "relu_fp16_spatz_params.h"
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
