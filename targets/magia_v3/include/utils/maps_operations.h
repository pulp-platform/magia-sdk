#ifndef MAPS_OPERATIONS_H
#define MAPS_OPERATIONS_H

#include "utils/maps_utils.h"
#include "utils/maps_operation_indexing.h"

#include "eventunit.h"
#include "redmule.h"
#include "utils/magia_spatz_utils.h"

#ifndef MAPS_HAS_ADD_SPATZ_TASK
#define MAPS_HAS_ADD_SPATZ_TASK 0u
#endif
#ifndef MAPS_HAS_MUL_SPATZ_TASK
#define MAPS_HAS_MUL_SPATZ_TASK 0u
#endif
#ifndef MAPS_HAS_MATMUL_SPATZ_TASK
#define MAPS_HAS_MATMUL_SPATZ_TASK 0u
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
    idma_controller_t *idma_ctrl;
    redmule_controller_t *redmule_ctrl;
    eu_controller_t *eu_ctrl;
    uint32_t kernel_abi_version;
    uint32_t task_bundle_abi_version;
    uint32_t spatz_binary_start;
    uint32_t add_fp16_task;
    uint32_t mul_bcast_fp16_task;
    uint32_t matmul_fp16_task;
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

static inline float maps_operation_param_f32(uint32_t bits)
{
    union { uint32_t bits; float value; } parameter = {.bits = bits};
    return parameter.value;
}

static inline uint32_t maps_operation_low_u16(uint32_t value)
{
    return value & 0xffffu;
}

static inline uint32_t maps_operation_high_u16(uint32_t value)
{
    return value >> 16;
}

static inline uint32_t maps_operation_global_offset(
    const tile_plan_t *plan, const subslice_desc_t *subslice, uint32_t axis)
{
    const slice_desc_t *slice = get_slice(plan, subslice->slice_id);
    return slice->full_offset[axis] + subslice->offset[axis];
}

static inline float maps_operation_sigmoid_f32(float value)
{
    const float minimum = maps_operation_f16_to_f32(
        maps_operation_f32_to_f16(-11.0f));
    const float clamped = value > minimum ? value : minimum;
    const float coefficient = maps_operation_f16_to_f32(
        maps_operation_f32_to_f16(1477.0f));
    const float bias = maps_operation_f16_to_f32(
        maps_operation_f32_to_f16(15360.0f));
    const uint16_t scaled = maps_operation_f32_to_f16(-clamped * coefficient);
    const uint16_t biased = maps_operation_f32_to_f16(
        maps_operation_f16_to_f32(scaled) + bias);
    const uint16_t exponential_bits =
        (uint16_t)maps_operation_f16_to_f32(biased);
    const float exponential = maps_operation_f16_to_f32(exponential_bits);
    const uint16_t denominator = maps_operation_f32_to_f16(1.0f + exponential);
    return 1.0f / maps_operation_f16_to_f32(denominator);
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
            ? maps_operation_f16_to_f32(rhs[rhs_index])
            : maps_operation_param_f32(op->params[0]);
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
        case OP_SIGMOID:
            result = maps_operation_sigmoid_f32(left);
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

static inline int maps_execute_im2col_scalar_f16(const tile_plan_t *plan,
                                                 const op_desc_t *op,
                                                 uint32_t slot)
{
    if (op->num_inputs != 1u || op->num_outputs != 1u ||
        op->inputs[0].rank != 4u || op->outputs[0].rank != 2u ||
        op->inputs[0].shape[0] != 1u)
        return -1;
    const uint8_t *input = (const uint8_t *)local_subslice_addr(
        plan, &op->inputs[0], slot);
    uint8_t *output = (uint8_t *)local_subslice_addr(
        plan, &op->outputs[0], slot);
    const uint32_t kernel_h = maps_operation_low_u16(op->params[0]);
    const uint32_t kernel_w = maps_operation_high_u16(op->params[0]);
    const uint32_t stride_h = maps_operation_low_u16(op->params[1]);
    const uint32_t stride_w = maps_operation_high_u16(op->params[1]);
    const uint32_t dilation_h = maps_operation_low_u16(op->params[2]);
    const uint32_t dilation_w = maps_operation_high_u16(op->params[2]);
    const uint32_t pad_top = maps_operation_low_u16(op->params[3]);
    const uint32_t pad_left = maps_operation_high_u16(op->params[3]);
    const uint32_t input_h = maps_operation_low_u16(op->params[5]);
    const uint32_t input_w = maps_operation_high_u16(op->params[5]);
    const uint32_t output_h = maps_operation_low_u16(op->params[6]);
    const uint32_t output_w = maps_operation_high_u16(op->params[6]);
    const uint32_t kernel_elements = kernel_h * kernel_w;
    if (kernel_elements == 0u || stride_h == 0u || stride_w == 0u ||
        output_h == 0u || output_w == 0u ||
        op->outputs[0].shape[1] % kernel_elements != 0u)
        return -2;

    const uint32_t input_origin_n = maps_operation_global_offset(
        plan, &op->inputs[0], 0u);
    const uint32_t input_origin_c = maps_operation_global_offset(
        plan, &op->inputs[0], 1u);
    const uint32_t input_origin_h = maps_operation_global_offset(
        plan, &op->inputs[0], 2u);
    const uint32_t input_origin_w = maps_operation_global_offset(
        plan, &op->inputs[0], 3u);
    const uint32_t channels_first = op->params[7] != 0u;
    const uint32_t output_origin_0 = maps_operation_global_offset(
        plan, &op->outputs[0], 0u);
    const uint32_t output_origin_1 = maps_operation_global_offset(
        plan, &op->outputs[0], 1u);
    for (uint32_t outer = 0u; outer < op->outputs[0].shape[0]; ++outer) {
        for (uint32_t inner = 0u; inner < op->outputs[0].shape[1]; ++inner) {
            const uint32_t global_row = channels_first
                ? output_origin_1 + inner : output_origin_0 + outer;
            const uint32_t global_column = channels_first
                ? output_origin_0 + outer : output_origin_1 + inner;
            const uint32_t input_origin[4] = {
                input_origin_n, input_origin_c, input_origin_h, input_origin_w};
            maps_local_nchw_t local;
            uint16_t value = 0u;
            if (maps_im2col_local_nchw(
                    global_row, global_column, output_h, output_w,
                    kernel_h, kernel_w, stride_h, stride_w,
                    dilation_h, dilation_w, pad_top, pad_left,
                    input_h, input_w, input_origin, op->inputs[0].shape,
                    &local)) {
                const uint32_t input_offset =
                    local.batch * op->inputs[0].strides_bytes[0] +
                    local.channel * op->inputs[0].strides_bytes[1] +
                    local.row * op->inputs[0].strides_bytes[2] +
                    local.column * op->inputs[0].strides_bytes[3];
                value = *(const uint16_t *)(input + input_offset);
            }
            const uint32_t output_offset =
                outer * op->outputs[0].strides_bytes[0] +
                inner * op->outputs[0].strides_bytes[1];
            *(uint16_t *)(output + output_offset) = value;
        }
    }
    return 0;
}

static inline int maps_execute_im2col_f16(
    const tile_plan_t *plan, const op_desc_t *op, uint32_t slot,
    maps_operation_runtime_t *runtime)
{
    if (op->params[7] == 0u)
        return maps_execute_im2col_scalar_f16(plan, op, slot);
    if (!runtime || !runtime->idma_ctrl || !runtime->eu_ctrl ||
        op->num_inputs != 1u || op->num_outputs != 1u ||
        op->inputs[0].rank != 4u || op->outputs[0].rank != 2u ||
        op->inputs[0].shape[0] != 1u ||
        op->inputs[0].elem_type != ELEM_F16 ||
        op->outputs[0].elem_type != ELEM_F16)
        return -1;

    const uint32_t input = local_subslice_addr(plan, &op->inputs[0], slot);
    const uint32_t output = local_subslice_addr(plan, &op->outputs[0], slot);
    const uint32_t kernel_h = maps_operation_low_u16(op->params[0]);
    const uint32_t kernel_w = maps_operation_high_u16(op->params[0]);
    const uint32_t stride_h = maps_operation_low_u16(op->params[1]);
    const uint32_t stride_w = maps_operation_high_u16(op->params[1]);
    const uint32_t dilation_h = maps_operation_low_u16(op->params[2]);
    const uint32_t dilation_w = maps_operation_high_u16(op->params[2]);
    const uint32_t pad_top = maps_operation_low_u16(op->params[3]);
    const uint32_t pad_left = maps_operation_high_u16(op->params[3]);
    const uint32_t input_h = maps_operation_low_u16(op->params[5]);
    const uint32_t input_w = maps_operation_high_u16(op->params[5]);
    const uint32_t output_w = maps_operation_high_u16(op->params[6]);
    const uint32_t local_k = op->outputs[0].shape[0];
    const uint32_t local_n = op->outputs[0].shape[1];

    if (kernel_h == 1u && kernel_w == 1u && stride_h == 1u &&
        stride_w == 1u && dilation_h == 1u && dilation_w == 1u &&
        pad_top == 0u && pad_left == 0u &&
        maps_operation_low_u16(op->params[4]) == 0u &&
        maps_operation_high_u16(op->params[4]) == 0u &&
        op->inputs[0].shape[1] == local_k &&
        op->inputs[0].shape[2] * op->inputs[0].shape[3] == local_n) {
        if (idma_memcpy_2d_ex(
                runtime->idma_ctrl, 1u, output, input,
                local_n * sizeof(uint16_t), op->outputs[0].strides_bytes[0],
                op->inputs[0].strides_bytes[1], local_k) != 0)
            return -2;
        return eu_idma_wait_o2a(runtime->eu_ctrl, MAPS_WAIT_MODE) ? 0 : -2;
    }

    if (output_w == 0u || local_n % output_w != 0u ||
        maps_operation_global_offset(plan, &op->outputs[0], 1u) % output_w != 0u)
        return maps_execute_im2col_scalar_f16(plan, op, slot);

    maps_operation_zero(output, local_k * local_n * sizeof(uint16_t));
    const uint32_t local_output_h = local_n / output_w;
    const uint32_t output_h_start =
        maps_operation_global_offset(plan, &op->outputs[0], 1u) / output_w;
    const uint32_t input_origin_c = maps_operation_global_offset(
        plan, &op->inputs[0], 1u);
    const uint32_t input_origin_h = maps_operation_global_offset(
        plan, &op->inputs[0], 2u);
    const uint32_t input_origin_w = maps_operation_global_offset(
        plan, &op->inputs[0], 3u);
    const uint32_t output_k_start = maps_operation_global_offset(
        plan, &op->outputs[0], 0u);
    const uint32_t kernel_elements = kernel_h * kernel_w;

    for (uint32_t local_kernel = 0u; local_kernel < local_k; ++local_kernel) {
        const uint32_t global_kernel = output_k_start + local_kernel;
        const uint32_t channel = global_kernel / kernel_elements;
        const uint32_t kernel_offset = global_kernel % kernel_elements;
        const uint32_t kernel_y = kernel_offset / kernel_w;
        const uint32_t kernel_x = kernel_offset % kernel_w;
        const int32_t first_input_x =
            (int32_t)(kernel_x * dilation_w) - (int32_t)pad_left;
        uint32_t valid_x_start = 0u;
        while (valid_x_start < output_w &&
               (int32_t)(valid_x_start * stride_w) + first_input_x < 0)
            ++valid_x_start;
        uint32_t valid_x_end = output_w;
        while (valid_x_end > valid_x_start &&
               (int32_t)((valid_x_end - 1u) * stride_w) + first_input_x >=
                   (int32_t)input_w)
            --valid_x_end;
        if (valid_x_start >= valid_x_end)
            continue;

        uint32_t valid_y_start = 0u;
        while (valid_y_start < local_output_h) {
            const int32_t input_y =
                (int32_t)((output_h_start + valid_y_start) * stride_h +
                          kernel_y * dilation_h) - (int32_t)pad_top;
            if (input_y >= 0)
                break;
            ++valid_y_start;
        }
        uint32_t valid_y_end = valid_y_start;
        while (valid_y_end < local_output_h) {
            const int32_t input_y =
                (int32_t)((output_h_start + valid_y_end) * stride_h +
                          kernel_y * dilation_h) - (int32_t)pad_top;
            if (input_y >= (int32_t)input_h)
                break;
            ++valid_y_end;
        }
        if (valid_y_start == valid_y_end)
            continue;

        const uint32_t global_input_y =
            (output_h_start + valid_y_start) * stride_h +
            kernel_y * dilation_h - pad_top;
        const uint32_t global_input_x =
            valid_x_start * stride_w + first_input_x;
        const uint32_t source = input +
            (channel - input_origin_c) * op->inputs[0].strides_bytes[1] +
            (global_input_y - input_origin_h) * op->inputs[0].strides_bytes[2] +
            (global_input_x - input_origin_w) * op->inputs[0].strides_bytes[3];
        const uint32_t destination = output +
            local_kernel * op->outputs[0].strides_bytes[0] +
            (valid_y_start * output_w + valid_x_start) *
                op->outputs[0].strides_bytes[1];
        if (stride_w == 1u) {
            if (idma_memcpy_2d_ex(
                    runtime->idma_ctrl, 1u, destination, source,
                    (valid_x_end - valid_x_start) * sizeof(uint16_t),
                    output_w * sizeof(uint16_t),
                    stride_h * op->inputs[0].strides_bytes[2],
                    valid_y_end - valid_y_start) != 0)
                return -2;
            if (!eu_idma_wait_o2a(runtime->eu_ctrl, MAPS_WAIT_MODE))
                return -2;
        } else {
            for (uint32_t y = valid_y_start; y < valid_y_end; ++y) {
                for (uint32_t x = valid_x_start; x < valid_x_end; ++x) {
                    const uint32_t element_source = source +
                        (y - valid_y_start) * stride_h *
                            op->inputs[0].strides_bytes[2] +
                        (x - valid_x_start) * stride_w * sizeof(uint16_t);
                    const uint32_t element_destination = output +
                        local_kernel * op->outputs[0].strides_bytes[0] +
                        (y * output_w + x) * op->outputs[0].strides_bytes[1];
                    if (idma_memcpy_1d(runtime->idma_ctrl, 1u,
                                       element_destination, element_source,
                                       sizeof(uint16_t)) != 0)
                        return -2;
                    if (!eu_idma_wait_o2a(runtime->eu_ctrl, MAPS_WAIT_MODE))
                        return -2;
                }
            }
        }
    }
    return 0;
}

static inline int maps_execute_transpose_f16(const tile_plan_t *plan,
                                              const op_desc_t *op,
                                              uint32_t slot)
{
    if (op->num_inputs != 1u || op->num_outputs != 1u ||
        op->inputs[0].rank != op->outputs[0].rank ||
        op->inputs[0].rank > MAPS_MAX_RANK)
        return -1;
    const uint8_t *input = (const uint8_t *)local_subslice_addr(
        plan, &op->inputs[0], slot);
    uint8_t *output = (uint8_t *)local_subslice_addr(
        plan, &op->outputs[0], slot);
    const uint32_t rank = op->outputs[0].rank;
    uint32_t input_origin[MAPS_MAX_RANK];
    uint32_t output_origin[MAPS_MAX_RANK];
    for (uint32_t axis = 0u; axis < rank; ++axis) {
        input_origin[axis] = maps_operation_global_offset(
            plan, &op->inputs[0], axis);
        output_origin[axis] = maps_operation_global_offset(
            plan, &op->outputs[0], axis);
    }
    return maps_local_transpose_f16(
        input, output, rank, op->inputs[0].shape,
        op->inputs[0].strides_bytes, input_origin, op->outputs[0].shape,
        op->outputs[0].strides_bytes, output_origin, op->params);
}

static inline int maps_execute_depthwise_conv_f16(const tile_plan_t *plan,
                                                   const op_desc_t *op,
                                                   uint32_t slot)
{
    if ((op->num_inputs != 2u && op->num_inputs != 3u) ||
        op->num_outputs != 1u || op->inputs[0].rank != 4u ||
        op->inputs[1].rank != 4u || op->outputs[0].rank != 4u)
        return -1;
    const uint16_t *input = (const uint16_t *)local_subslice_addr(
        plan, &op->inputs[0], slot);
    const uint16_t *weight = (const uint16_t *)local_subslice_addr(
        plan, &op->inputs[1], slot);
    const uint16_t *bias = op->num_inputs == 3u
        ? (const uint16_t *)local_subslice_addr(plan, &op->inputs[2], slot)
        : 0;
    uint16_t *output = (uint16_t *)local_subslice_addr(
        plan, &op->outputs[0], slot);
    const uint32_t stride_h = op->params[0];
    const uint32_t stride_w = op->params[1];
    const uint32_t dilation_h = op->params[2];
    const uint32_t dilation_w = op->params[3];
    const uint32_t pad_top = op->params[4];
    const uint32_t pad_left = op->params[5];
    const uint32_t channel_multiplier = op->params[6];
    if (stride_h == 0u || stride_w == 0u || channel_multiplier == 0u)
        return -2;
    const uint32_t input_origin_h = maps_operation_global_offset(
        plan, &op->inputs[0], 2u);
    const uint32_t input_origin_w = maps_operation_global_offset(
        plan, &op->inputs[0], 3u);
    const uint32_t output_origin_h = maps_operation_global_offset(
        plan, &op->outputs[0], 2u);
    const uint32_t output_origin_w = maps_operation_global_offset(
        plan, &op->outputs[0], 3u);
    const uint32_t weight_origin_c = maps_operation_global_offset(
        plan, &op->inputs[1], 0u);
    const uint32_t input_origin_c = maps_operation_global_offset(
        plan, &op->inputs[0], 1u);
    return maps_local_depthwise_conv_f16(
        input, weight, bias, output, op->inputs[0].shape,
        op->inputs[1].shape, op->outputs[0].shape, input_origin_c,
        input_origin_h, input_origin_w, weight_origin_c, output_origin_h,
        output_origin_w, stride_h, stride_w, dilation_h, dilation_w,
        pad_top, pad_left, channel_multiplier);
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
#if MAPS_HAS_MATMUL_SPATZ_TASK
        return maps_execute_matmul_spatz(plan, op, slot, runtime);
#else
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
#endif
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
    case OP_SIGMOID:
    case OP_SUB:
    case OP_DIV:
        return maps_execute_elementwise_f16(plan, op, slot);
    case OP_MUL:
#if MAPS_HAS_MUL_SPATZ_TASK
        return maps_execute_mul_spatz(plan, op, slot, runtime);
#else
        return maps_execute_elementwise_f16(plan, op, slot);
#endif
    case OP_REDUCE_MAX:
    case OP_REDUCE_SUM:
        return maps_execute_reduce_f16(plan, op, slot);
    case OP_ALL_REDUCE_MAX:
    case OP_ALL_REDUCE_SUM:
        return maps_execute_all_reduce(plan, op, slot);
    case OP_SPLIT:
        return maps_execute_split_f16(plan, op, slot);
    case OP_IM2COL:
        return maps_execute_im2col_f16(plan, op, slot, runtime);
    case OP_TRANSPOSE:
        return maps_execute_transpose_f16(plan, op, slot);
    case OP_DEPTHWISE_CONV:
        return maps_execute_depthwise_conv_f16(plan, op, slot);
    case OP_OUTPUT_REFORMAT:
        return maps_execute_output_reformat_1x1_f16(plan, op, slot);
    default:
        return maps_execute_builtin_op(plan, op, slot);
    }
}

#endif
