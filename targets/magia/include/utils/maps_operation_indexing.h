#ifndef MAPS_OPERATION_INDEXING_H
#define MAPS_OPERATION_INDEXING_H

#include <stdint.h>

typedef struct {
    uint32_t batch;
    uint32_t channel;
    uint32_t row;
    uint32_t column;
} maps_local_nchw_t;

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

static inline int maps_rebase_coordinate(uint32_t global_coordinate,
                                         uint32_t local_origin,
                                         uint32_t local_size,
                                         uint32_t *local_coordinate)
{
    if (global_coordinate < local_origin ||
        global_coordinate - local_origin >= local_size)
        return 0;
    *local_coordinate = global_coordinate - local_origin;
    return 1;
}

static inline int maps_im2col_local_nchw(
    uint32_t global_row, uint32_t global_column,
    uint32_t output_h, uint32_t output_w,
    uint32_t kernel_h, uint32_t kernel_w,
    uint32_t stride_h, uint32_t stride_w,
    uint32_t dilation_h, uint32_t dilation_w,
    uint32_t pad_top, uint32_t pad_left,
    uint32_t input_h, uint32_t input_w,
    const uint32_t input_origin[4], const uint32_t input_shape[4],
    maps_local_nchw_t *local)
{
    const uint32_t kernel_elements = kernel_h * kernel_w;
    const uint32_t batch = global_row / (output_h * output_w);
    const uint32_t spatial = global_row % (output_h * output_w);
    const uint32_t output_row = spatial / output_w;
    const uint32_t output_column = spatial % output_w;
    const uint32_t channel = global_column / kernel_elements;
    const uint32_t kernel_index = global_column % kernel_elements;
    const int32_t input_row = (int32_t)(output_row * stride_h) -
        (int32_t)pad_top +
        (int32_t)((kernel_index / kernel_w) * dilation_h);
    const int32_t input_column = (int32_t)(output_column * stride_w) -
        (int32_t)pad_left +
        (int32_t)((kernel_index % kernel_w) * dilation_w);

    if (input_row < 0 || input_column < 0 ||
        (uint32_t)input_row >= input_h || (uint32_t)input_column >= input_w ||
        !maps_rebase_coordinate(batch, input_origin[0], input_shape[0],
                                &local->batch) ||
        !maps_rebase_coordinate(channel, input_origin[1], input_shape[1],
                                &local->channel) ||
        !maps_rebase_coordinate((uint32_t)input_row, input_origin[2],
                                input_shape[2], &local->row) ||
        !maps_rebase_coordinate((uint32_t)input_column, input_origin[3],
                                input_shape[3], &local->column))
        return 0;
    return 1;
}

static inline int maps_depthwise_local_input_channel(
    uint32_t global_output_channel, uint32_t channel_multiplier,
    uint32_t input_origin, uint32_t input_channels,
    uint32_t *local_input_channel)
{
    return maps_rebase_coordinate(global_output_channel / channel_multiplier,
                                  input_origin, input_channels,
                                  local_input_channel);
}

static inline int maps_local_transpose_f16(
    const uint8_t *input, uint8_t *output, uint32_t rank,
    const uint32_t *input_shape, const uint32_t *input_strides,
    const uint32_t *input_origin, const uint32_t *output_shape,
    const uint32_t *output_strides, const uint32_t *output_origin,
    const uint32_t *permutation)
{
    uint32_t elements = 1u;
    for (uint32_t axis = 0u; axis < rank; ++axis)
        elements *= output_shape[axis];
    for (uint32_t index = 0u; index < elements; ++index) {
        uint32_t remaining = index;
        uint32_t output_coordinates[8];
        uint32_t input_coordinates[8];
        for (uint32_t reversed = rank; reversed > 0u; --reversed) {
            const uint32_t axis = reversed - 1u;
            output_coordinates[axis] = remaining % output_shape[axis];
            remaining /= output_shape[axis];
        }
        for (uint32_t axis = 0u; axis < rank; ++axis) {
            const uint32_t input_axis = permutation[axis];
            if (input_axis >= rank)
                return -1;
            if (!maps_rebase_coordinate(
                    output_origin[axis] + output_coordinates[axis],
                    input_origin[input_axis], input_shape[input_axis],
                    &input_coordinates[input_axis]))
                return -2;
        }
        uint32_t input_offset = 0u;
        uint32_t output_offset = 0u;
        for (uint32_t axis = 0u; axis < rank; ++axis) {
            input_offset += input_coordinates[axis] * input_strides[axis];
            output_offset += output_coordinates[axis] * output_strides[axis];
        }
        *(uint16_t *)(output + output_offset) =
            *(const uint16_t *)(input + input_offset);
    }
    return 0;
}

static inline int maps_local_depthwise_conv_f16(
    const uint16_t *input, const uint16_t *weight, const uint16_t *bias,
    uint16_t *output, const uint32_t input_shape[4],
    const uint32_t weight_shape[4], const uint32_t output_shape[4],
    uint32_t input_origin_c, uint32_t input_origin_h,
    uint32_t input_origin_w, uint32_t weight_origin_c,
    uint32_t output_origin_h, uint32_t output_origin_w,
    uint32_t stride_h, uint32_t stride_w, uint32_t dilation_h,
    uint32_t dilation_w, uint32_t pad_top, uint32_t pad_left,
    uint32_t channel_multiplier)
{
    for (uint32_t n = 0u; n < output_shape[0]; ++n)
        for (uint32_t channel = 0u; channel < output_shape[1]; ++channel) {
            uint32_t input_channel;
            if (!maps_depthwise_local_input_channel(
                    weight_origin_c + channel, channel_multiplier,
                    input_origin_c, input_shape[1], &input_channel))
                return -1;
            for (uint32_t oh = 0u; oh < output_shape[2]; ++oh)
                for (uint32_t ow = 0u; ow < output_shape[3]; ++ow) {
                    float sum = bias ? maps_operation_f16_to_f32(bias[channel]) : 0.0f;
                    for (uint32_t kh = 0u; kh < weight_shape[2]; ++kh)
                        for (uint32_t kw = 0u; kw < weight_shape[3]; ++kw) {
                            const int32_t global_h =
                                (int32_t)((output_origin_h + oh) * stride_h) -
                                (int32_t)pad_top + (int32_t)(kh * dilation_h);
                            const int32_t global_w =
                                (int32_t)((output_origin_w + ow) * stride_w) -
                                (int32_t)pad_left + (int32_t)(kw * dilation_w);
                            uint32_t local_h;
                            uint32_t local_w;
                            if (global_h < 0 || global_w < 0 ||
                                !maps_rebase_coordinate(
                                    (uint32_t)global_h, input_origin_h,
                                    input_shape[2], &local_h) ||
                                !maps_rebase_coordinate(
                                    (uint32_t)global_w, input_origin_w,
                                    input_shape[3], &local_w))
                                continue;
                            const uint32_t input_index =
                                ((n * input_shape[1] + input_channel) *
                                 input_shape[2] + local_h) * input_shape[3] +
                                local_w;
                            const uint32_t weight_index =
                                (channel * weight_shape[2] + kh) *
                                weight_shape[3] + kw;
                            sum += maps_operation_f16_to_f32(input[input_index]) *
                                maps_operation_f16_to_f32(weight[weight_index]);
                        }
                    const uint32_t output_index =
                        ((n * output_shape[1] + channel) * output_shape[2] + oh) *
                            output_shape[3] + ow;
                    output[output_index] = maps_operation_f32_to_f16(sum);
                }
        }
    return 0;
}

#endif
