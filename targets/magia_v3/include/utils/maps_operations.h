#ifndef MAPS_OPERATIONS_H
#define MAPS_OPERATIONS_H

#include "utils/maps_utils.h"

#include "eventunit.h"
#include "redmule.h"
#include "utils/magia_spatz_utils.h"

#ifndef MAPS_HAS_ADD_SPATZ_TASK
#define MAPS_HAS_ADD_SPATZ_TASK 0u
#endif
#ifndef MAPS_HAS_RELU_SPATZ_TASK
#define MAPS_HAS_RELU_SPATZ_TASK 0u
#endif
#ifndef MAPS_HAS_SOFTMAX_EXP_SPATZ_TASK
#define MAPS_HAS_SOFTMAX_EXP_SPATZ_TASK 0u
#endif
#ifndef MAPS_HAS_GROUP_REDUCE_SPATZ_TASK
#define MAPS_HAS_GROUP_REDUCE_SPATZ_TASK 0u
#endif
#ifndef MAPS_HAS_GROUP_CENTERED_REDUCE_SPATZ_TASK
#define MAPS_HAS_GROUP_CENTERED_REDUCE_SPATZ_TASK 0u
#endif
#ifndef MAPS_HAS_GROUP_NORMALIZE_SPATZ_TASK
#define MAPS_HAS_GROUP_NORMALIZE_SPATZ_TASK 0u
#endif

#ifndef MAPS_KERNEL_ABI_VERSION
#define MAPS_KERNEL_ABI_VERSION 2u
#endif
#ifndef MAPS_TASK_BUNDLE_ABI_VERSION
#define MAPS_TASK_BUNDLE_ABI_VERSION 2u
#endif
#define MAPS_OPERATION_TASK_SCRATCH_OFFSET 0xC0000u
#define MAPS_OPERATION_TASK_SCRATCH_BYTES 0x10000u

typedef struct {
    redmule_controller_t *redmule_ctrl;
    eu_controller_t *eu_ctrl;
    uint32_t kernel_abi_version;
    uint32_t task_bundle_abi_version;
    uint32_t spatz_binary_start;
    uint32_t add_fp16_task;
    uint32_t relu_fp16_task;
    uint32_t softmax_exp_fp16_task;
    uint32_t group_reduce_fp16_task;
    uint32_t group_centered_reduce_fp16_task;
    uint32_t group_normalize_fp16_task;
    void *spatz_params;
    uint32_t spatz_params_bytes;
    uint32_t spatz_initialized;
} maps_operation_runtime_t;

static inline int maps_operation_runtime_init(maps_operation_runtime_t *runtime)
{
    if (!runtime)
        return -1;
    if (runtime->spatz_binary_start == 0u)
        return 0;
    if (runtime->kernel_abi_version != MAPS_KERNEL_ABI_VERSION ||
        runtime->task_bundle_abi_version != MAPS_TASK_BUNDLE_ABI_VERSION)
        return -2;
    if (!runtime->eu_ctrl || !runtime->spatz_params ||
        runtime->spatz_params_bytes == 0u)
        return -3;
    if (!runtime->spatz_initialized) {
        eu_spatz_init(runtime->eu_ctrl, 0u);
        spatz_init(runtime->spatz_binary_start);
        runtime->spatz_initialized = 1u;
    }
    return 0;
}

typedef enum {
    MAPS_COLLECTIVE_ARRIVAL = 0u,
    MAPS_COLLECTIVE_RELEASE = 1u,
} maps_collective_phase_t;

static inline float maps_operation_f16_to_f32(uint16_t value)
{
    const uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    uint32_t mantissa = value & 0x03ffu;
    int32_t exponent = (int32_t)((value >> 10) & 0x1fu);
    union { uint32_t bits; float value; } result;

    if (exponent == 0) {
        if (mantissa == 0u) {
            result.bits = sign;
            return result.value;
        }
        exponent = 1;
        while ((mantissa & 0x0400u) == 0u) {
            mantissa <<= 1;
            --exponent;
        }
        mantissa &= 0x03ffu;
    } else if (exponent == 31) {
        result.bits = sign | 0x7f800000u | (mantissa << 13);
        return result.value;
    }
    result.bits = sign | ((uint32_t)(exponent + 112) << 23) |
                  (mantissa << 13);
    return result.value;
}

static inline uint16_t maps_operation_f32_to_f16(float value)
{
    union { float value; uint32_t bits; } input = {.value = value};
    const uint32_t sign = (input.bits >> 16) & 0x8000u;
    int32_t exponent = (int32_t)((input.bits >> 23) & 0xffu) - 112;
    uint32_t mantissa = input.bits & 0x007fffffu;

    if (((input.bits >> 23) & 0xffu) == 0xffu)
        return (uint16_t)(sign | (mantissa != 0u ? 0x7e00u : 0x7c00u));
    if (exponent >= 31)
        return (uint16_t)(sign | 0x7c00u);
    if (exponent <= 0) {
        if (exponent < -10)
            return (uint16_t)sign;
        mantissa |= 0x00800000u;
        const uint32_t shift = (uint32_t)(14 - exponent);
        uint32_t result = mantissa >> shift;
        const uint32_t halfway = 1u << (shift - 1u);
        if ((mantissa & halfway) &&
            ((mantissa & (halfway - 1u)) || (result & 1u)))
            ++result;
        return (uint16_t)(sign | result);
    }
    mantissa += 0x00000fffu + ((mantissa >> 13) & 1u);
    if (mantissa & 0x00800000u) {
        mantissa = 0u;
        if (++exponent >= 31)
            return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exponent << 10) |
                      (mantissa >> 13));
}

static inline float maps_operation_exp_f32(float value)
{
    if (value <= -80.0f)
        return 0.0f;
    if (value >= 80.0f)
        value = 80.0f;
    const float scaled = value * 1.4426950408889634f;
    int32_t exponent = (int32_t)scaled;
    if (value < 0.0f && (float)exponent != scaled)
        --exponent;
    const float remainder = value - (float)exponent * 0.6931471805599453f;
    const float square = remainder * remainder;
    const float polynomial = 1.0f + remainder + square *
        (0.5f + remainder * (0.1666666716f + remainder *
        (0.0416666679f + remainder * (0.0083333338f +
         remainder * 0.0013888889f))));
    union { uint32_t bits; float value; } scale = {
        .bits = (uint32_t)(exponent + 127) << 23
    };
    return polynomial * scale.value;
}

static inline uint32_t maps_operation_elems(const subslice_desc_t *value)
{
    return maps_shape_elems(value->rank, value->shape);
}

static inline uint32_t maps_operation_broadcast_index(
    const subslice_desc_t *output, const subslice_desc_t *input,
    uint32_t output_index)
{
    uint32_t input_index = 0u;
    uint32_t input_stride = 1u;
    uint32_t remaining = output_index;
    uint32_t output_axis = output->rank;
    uint32_t input_axis = input->rank;

    while (output_axis > 0u) {
        --output_axis;
        const uint32_t coordinate = remaining % output->shape[output_axis];
        remaining /= output->shape[output_axis];
        if (input_axis > 0u) {
            --input_axis;
            if (input->shape[input_axis] != 1u)
                input_index += coordinate * input_stride;
            input_stride *= input->shape[input_axis];
        }
    }
    return input_index;
}

static inline int maps_execute_elementwise_f16(const tile_plan_t *plan,
                                                const op_desc_t *op,
                                                uint32_t slot)
{
    if (op->num_inputs == 0u || op->num_outputs == 0u ||
        op->outputs[0].elem_type != ELEM_F16)
        return -1;

    const uint16_t *lhs = (const uint16_t *)local_subslice_addr(
        plan, &op->inputs[0], slot);
    const uint16_t *rhs = op->num_inputs > 1u
        ? (const uint16_t *)local_subslice_addr(plan, &op->inputs[1], slot)
        : 0;
    uint16_t *output = (uint16_t *)local_subslice_addr(
        plan, &op->outputs[0], slot);
    const uint32_t output_elems = maps_operation_elems(&op->outputs[0]);
    const uint32_t rhs_elems = rhs ? maps_operation_elems(&op->inputs[1]) : 0u;

    for (uint32_t index = 0u; index < output_elems; ++index) {
        const uint32_t lhs_index = maps_operation_broadcast_index(
            &op->outputs[0], &op->inputs[0], index);
        const float left = maps_operation_f16_to_f32(lhs[lhs_index]);
        const uint32_t rhs_index = rhs_elems == output_elems ? index :
            maps_operation_broadcast_index(
                &op->outputs[0], &op->inputs[1], index);
        const float right = rhs
            ? maps_operation_f16_to_f32(rhs[rhs_index]) : 0.0f;
        float result;

        switch (op->kind) {
        case OP_RELU:
            result = left > 0.0f ? left : 0.0f;
            break;
        case OP_NEG:
            result = -left;
            break;
        case OP_EXP:
            result = maps_operation_exp_f32(left);
            break;
        case OP_SUB:
            result = left - right;
            break;
        case OP_DIV:
            result = left / right;
            break;
        case OP_MUL:
            result = left * right;
            break;
        default:
            return -1;
        }
        output[index] = maps_operation_f32_to_f16(result);
    }
    return 0;
}

static inline int maps_execute_reduce_f16(const tile_plan_t *plan,
                                          const op_desc_t *op,
                                          uint32_t slot)
{
    if (op->num_inputs == 0u || op->num_outputs == 0u ||
        op->inputs[0].elem_type != ELEM_F16 ||
        op->outputs[0].elem_type != ELEM_F16 ||
        op->params[0] != op->inputs[0].rank - 1u)
        return -1;

    const uint16_t *input = (const uint16_t *)local_subslice_addr(
        plan, &op->inputs[0], slot);
    uint16_t *output = (uint16_t *)local_subslice_addr(
        plan, &op->outputs[0], slot);
    const uint32_t columns =
        op->inputs[0].shape[op->inputs[0].rank - 1u];
    const uint32_t rows = maps_operation_elems(&op->inputs[0]) / columns;

    for (uint32_t row = 0u; row < rows; ++row) {
        float result = maps_operation_f16_to_f32(input[row * columns]);
        for (uint32_t column = 1u; column < columns; ++column) {
            const float value = maps_operation_f16_to_f32(
                input[row * columns + column]);
            result = op->kind == OP_REDUCE_SUM
                ? result + value : (value > result ? value : result);
        }
        output[row] = maps_operation_f32_to_f16(result);
    }
    return 0;
}

static inline volatile uint32_t *maps_collective_flag(
    const tile_plan_t *plan, uint32_t hartid, uint32_t flag_index)
{
    if (flag_index >= plan->ready_flags_count)
        maps_trap();
    return (volatile uint32_t *)(remote_tile_l1_base(hartid) +
                                 plan->ready_flags_base +
                                 flag_index * sizeof(uint32_t));
}

static inline uint32_t maps_collective_flag_index(
    const collective_desc_t *collective, uint32_t slot,
    maps_collective_phase_t phase, uint32_t participant_index)
{
    return collective->flag_base +
        ((slot * 2u + phase) * collective->num_participants) +
        participant_index;
}

static inline void maps_collective_publish(const tile_plan_t *plan,
                                           uint32_t hartid,
                                           uint32_t flag_index)
{
    __asm__ volatile("fence rw, w" ::: "memory");
    *maps_collective_flag(plan, hartid, flag_index) = 1u;
    __asm__ volatile("fence w, w" ::: "memory");
}

static inline void maps_collective_wait(const tile_plan_t *plan,
                                        uint32_t flag_index)
{
    volatile uint32_t *flag = maps_collective_flag(
        plan, plan->hartid, flag_index);
    while (*flag != 1u)
        __asm__ volatile("" ::: "memory");
    __asm__ volatile("fence r, rw" ::: "memory");
    *flag = 0u;
    __asm__ volatile("fence rw, rw" ::: "memory");
}

static inline int maps_execute_all_reduce(const tile_plan_t *plan,
                                          const op_desc_t *op,
                                          uint32_t slot)
{
    const collective_desc_t *collective = &op->collective;
    if (collective->num_participants == 0u ||
        collective->num_participants > MAPS_MAX_COLLECTIVE_PARTICIPANTS ||
        slot >= collective->num_token_slots ||
        op->num_inputs == 0u || op->num_outputs == 0u ||
        op->inputs[0].elem_type != ELEM_F16 ||
        op->outputs[0].elem_type != ELEM_F16)
        return -1;

    uint32_t local_index = collective->num_participants;
    for (uint32_t index = 0u; index < collective->num_participants; ++index)
        if (collective->participants[index].hartid == plan->hartid)
            local_index = index;
    if (local_index == collective->num_participants)
        return -1;

    const uint32_t root_hartid = collective->participants[0].hartid;
    if (plan->hartid != root_hartid) {
        maps_collective_publish(
            plan, root_hartid,
            maps_collective_flag_index(
                collective, slot, MAPS_COLLECTIVE_ARRIVAL, local_index));
        maps_collective_wait(
            plan,
            maps_collective_flag_index(
                collective, slot, MAPS_COLLECTIVE_RELEASE, local_index));
        return 0;
    }

    for (uint32_t index = 1u; index < collective->num_participants; ++index)
        maps_collective_wait(
            plan,
            maps_collective_flag_index(
                collective, slot, MAPS_COLLECTIVE_ARRIVAL, index));

    const uint32_t elements = maps_operation_elems(&op->outputs[0]);
    for (uint32_t element = 0u; element < elements; ++element) {
        float result = 0.0f;
        for (uint32_t index = 0u; index < collective->num_participants; ++index) {
            const collective_participant_desc_t *participant =
                &collective->participants[index];
            const uint16_t *input = (const uint16_t *)(
                remote_tile_l1_base(participant->hartid) +
                participant->l1_data_base_offset +
                participant->input_l1_offset_bytes +
                slot * participant->input_slot_bytes);
            const float value = maps_operation_f16_to_f32(input[element]);
            if (index == 0u || op->kind == OP_ALL_REDUCE_MAX)
                result = index == 0u || value > result ? value : result;
            else
                result += value;
        }
        const uint16_t reduced = maps_operation_f32_to_f16(result);
        for (uint32_t index = 0u; index < collective->num_participants; ++index) {
            const collective_participant_desc_t *participant =
                &collective->participants[index];
            volatile uint16_t *output = (volatile uint16_t *)(
                remote_tile_l1_base(participant->hartid) +
                participant->l1_data_base_offset +
                participant->output_l1_offset_bytes +
                slot * participant->output_slot_bytes);
            output[element] = reduced;
        }
    }

    __asm__ volatile("fence rw, rw" ::: "memory");
    for (uint32_t index = 1u; index < collective->num_participants; ++index)
        maps_collective_publish(
            plan, collective->participants[index].hartid,
            maps_collective_flag_index(
                collective, slot, MAPS_COLLECTIVE_RELEASE, index));
    return 0;
}

static inline void maps_operation_zero(uint32_t address, uint32_t bytes)
{
    volatile uint8_t *output = (volatile uint8_t *)address;
    for (uint32_t index = 0u; index < bytes; ++index)
        output[index] = 0u;
}

static inline int maps_execute_split_f16(const tile_plan_t *plan,
                                         const op_desc_t *op,
                                         uint32_t slot)
{
    if (op->num_inputs != 1u || op->num_outputs == 0u ||
        op->num_outputs > MAPS_MAX_OP_OUTPUTS || op->params[0] != 1u)
        return -1;
    const uint16_t *input = (const uint16_t *)local_subslice_addr(
        plan, &op->inputs[0], slot);
    uint32_t offset = 0u;
    for (uint32_t output_index = 0u; output_index < op->num_outputs;
         ++output_index) {
        const uint32_t elements = maps_operation_elems(&op->outputs[output_index]);
        uint16_t *output = (uint16_t *)local_subslice_addr(
            plan, &op->outputs[output_index], slot);
        for (uint32_t index = 0u; index < elements; ++index)
            output[index] = input[offset + index];
        offset += elements;
    }
    return offset == maps_operation_elems(&op->inputs[0]) ? 0 : -2;
}

static inline int maps_execute_im2col_1x1_f16(const tile_plan_t *plan,
                                              const op_desc_t *op,
                                              uint32_t slot)
{
    if (op->num_inputs != 1u || op->num_outputs != 1u ||
        op->inputs[0].rank != 4u || op->outputs[0].rank != 2u ||
        op->inputs[0].shape[0] != 1u)
        return -1;
    const uint16_t *input = (const uint16_t *)local_subslice_addr(
        plan, &op->inputs[0], slot);
    uint16_t *output = (uint16_t *)local_subslice_addr(
        plan, &op->outputs[0], slot);
    const uint32_t channels = op->inputs[0].shape[1];
    const uint32_t spatial =
        op->inputs[0].shape[2] * op->inputs[0].shape[3];
    if (op->outputs[0].shape[0] != spatial ||
        op->outputs[0].shape[1] != channels)
        return -2;
    for (uint32_t row = 0u; row < spatial; ++row)
        for (uint32_t channel = 0u; channel < channels; ++channel)
            output[row * channels + channel] = input[channel * spatial + row];
    return 0;
}

static inline int maps_execute_output_reformat_1x1_f16(
    const tile_plan_t *plan, const op_desc_t *op, uint32_t slot)
{
    if (op->num_inputs != 1u || op->num_outputs != 1u ||
        op->inputs[0].rank != 2u || op->outputs[0].rank != 4u ||
        op->outputs[0].shape[0] != 1u)
        return -1;
    const uint16_t *input = (const uint16_t *)local_subslice_addr(
        plan, &op->inputs[0], slot);
    uint16_t *output = (uint16_t *)local_subslice_addr(
        plan, &op->outputs[0], slot);
    const uint32_t channels = op->outputs[0].shape[1];
    const uint32_t spatial =
        op->outputs[0].shape[2] * op->outputs[0].shape[3];
    if (op->inputs[0].shape[0] != spatial ||
        op->inputs[0].shape[1] != channels)
        return -2;
    for (uint32_t channel = 0u; channel < channels; ++channel)
        for (uint32_t row = 0u; row < spatial; ++row)
            output[channel * spatial + row] = input[row * channels + channel];
    return 0;
}

#include "utils/maps_spatz_tasks.h"

static inline int maps_execute_operation(const tile_plan_t *plan,
                                         const op_desc_t *op,
                                         uint32_t slot,
                                         maps_operation_runtime_t *runtime)
{
    switch (op->kind) {
    case OP_MATMUL: {
        if (!runtime || op->num_inputs < 2u || op->num_outputs == 0u ||
            op->outputs[0].elem_type != ELEM_F16)
            return -1;
        const uint32_t output = local_subslice_addr(
            plan, &op->outputs[0], slot);
        maps_operation_zero(
            output, op->params[0] * op->params[2] * sizeof(float16));
        if (redmule_gemm(
                runtime->redmule_ctrl,
                local_subslice_addr(plan, &op->inputs[0], slot),
                local_subslice_addr(plan, &op->inputs[1], slot), output,
                (uint16_t)op->params[0], (uint16_t)op->params[1],
                (uint16_t)op->params[2]) != 0)
            return -1;
        eu_redmule_wait(runtime->eu_ctrl, MAPS_WAIT_MODE);
        if (op->num_inputs == 3u) {
            uint16_t *values = (uint16_t *)output;
            const uint16_t *bias = (const uint16_t *)local_subslice_addr(
                plan, &op->inputs[2], slot);
            for (uint32_t row = 0u; row < op->params[0]; ++row)
                for (uint32_t column = 0u; column < op->params[2]; ++column) {
                    const uint32_t index = row * op->params[2] + column;
                    values[index] = maps_operation_f32_to_f16(
                        maps_operation_f16_to_f32(values[index]) +
                        maps_operation_f16_to_f32(bias[column]));
                }
        }
        return 0;
    }
    case OP_COPY:
    case OP_RESHAPE:
        return maps_execute_copy_op(plan, op, slot);
#if MAPS_HAS_ADD_SPATZ_TASK
    case OP_ADD:
        return maps_execute_add_spatz(plan, op, slot, runtime);
#endif
#if MAPS_HAS_RELU_SPATZ_TASK
    case OP_RELU:
        return maps_execute_relu_spatz(plan, op, slot, runtime);
#endif
#if MAPS_HAS_SOFTMAX_EXP_SPATZ_TASK
    case OP_SOFTMAX_EXP:
        return maps_execute_softmax_exp_spatz(plan, op, slot, runtime);
#endif
#if MAPS_HAS_GROUP_REDUCE_SPATZ_TASK
    case OP_GROUP_REDUCE:
        return maps_execute_group_reduce_spatz(plan, op, slot, runtime);
#endif
#if MAPS_HAS_GROUP_CENTERED_REDUCE_SPATZ_TASK
    case OP_GROUP_CENTERED_REDUCE:
        return maps_execute_group_centered_reduce_spatz(
            plan, op, slot, runtime);
#endif
#if MAPS_HAS_GROUP_NORMALIZE_SPATZ_TASK
    case OP_GROUP_NORMALIZE:
        return maps_execute_group_normalize_spatz(plan, op, slot, runtime);
#endif
    case OP_NEG:
    case OP_EXP:
    case OP_SUB:
    case OP_DIV:
    case OP_MUL:
        return maps_execute_elementwise_f16(plan, op, slot);
    case OP_REDUCE_MAX:
    case OP_REDUCE_SUM:
        return maps_execute_reduce_f16(plan, op, slot);
    case OP_ALL_REDUCE_MAX:
    case OP_ALL_REDUCE_SUM:
        return maps_execute_all_reduce(plan, op, slot);
    case OP_SPLIT:
        return maps_execute_split_f16(plan, op, slot);
    case OP_IM2COL:
        return maps_execute_im2col_1x1_f16(plan, op, slot);
    case OP_OUTPUT_REFORMAT:
        return maps_execute_output_reformat_1x1_f16(plan, op, slot);
    default:
        return maps_execute_builtin_op(plan, op, slot);
    }
}

#endif
