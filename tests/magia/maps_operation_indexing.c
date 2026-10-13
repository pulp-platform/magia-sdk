#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "utils/maps_operation_indexing.h"

static void test_im2col_spatial_shard(void)
{
    const uint32_t origin[4] = {0, 0, 99, 0};
    const uint32_t shape[4] = {1, 3, 21, 256};
    uint32_t nonzero = 0;

    for (uint32_t row = 0; row < 1280; ++row)
        for (uint32_t column = 0; column < 27; ++column) {
            maps_local_nchw_t local;
            const int valid = maps_im2col_local_nchw(
                6400 + row, column, 128, 128, 3, 3, 2, 2, 1, 1, 1, 1,
                256, 256, origin, shape, &local);
            const uint32_t global_row = 6400 + row;
            const uint32_t oh = (global_row % (128 * 128)) / 128;
            const uint32_t ow = global_row % 128;
            const uint32_t kernel = column % 9;
            const int32_t ih = (int32_t)(oh * 2) - 1 + (int32_t)(kernel / 3);
            const int32_t iw = (int32_t)(ow * 2) - 1 + (int32_t)(kernel % 3);
            const int expected = ih >= 99 && ih < 120 && iw >= 0 && iw < 256;
            assert(valid == expected);
            if (valid) {
                assert(local.batch == 0);
                assert(local.channel == column / 9);
                assert(local.row == (uint32_t)ih - 99);
                assert(local.column == (uint32_t)iw);
                nonzero += 1;
            }
        }
    assert(nonzero > 0);
}

static void test_rank6_transpose_nonzero_origin(void)
{
    const uint32_t permutation[6] = {0, 1, 4, 2, 5, 3};
    const uint32_t input_origin[6] = {0, 0, 0, 1, 0, 0};
    const uint32_t output_origin[6] = {0, 0, 0, 0, 0, 1};
    const uint32_t input_shape[6] = {1, 2, 2, 2, 2, 2};
    const uint32_t output_shape[6] = {1, 2, 2, 2, 2, 2};
    const uint32_t strides[6] = {64, 32, 16, 8, 4, 2};
    uint16_t input[32];
    uint16_t output[32];

    for (uint32_t index = 0; index < 32; ++index)
        input[index] = (uint16_t)(1000 + index);
    assert(maps_local_transpose_f16(
               (const uint8_t *)input, (uint8_t *)output, 6, input_shape,
               strides, input_origin, output_shape, strides, output_origin,
               permutation) == 0);
    for (uint32_t index = 0; index < 32; ++index) {
        uint32_t remaining = index;
        uint32_t output_coordinate[6];
        uint32_t input_coordinate[6];
        for (uint32_t reversed = 6; reversed > 0; --reversed) {
            const uint32_t axis = reversed - 1;
            output_coordinate[axis] = remaining % output_shape[axis];
            remaining /= output_shape[axis];
        }
        for (uint32_t axis = 0; axis < 6; ++axis) {
            const uint32_t input_axis = permutation[axis];
            assert(maps_rebase_coordinate(
                output_origin[axis] + output_coordinate[axis],
                input_origin[input_axis], input_shape[input_axis],
                &input_coordinate[input_axis]));
        }
        uint32_t input_index = 0;
        uint32_t stride = 1;
        for (uint32_t reversed = 6; reversed > 0; --reversed) {
            const uint32_t axis = reversed - 1;
            input_index += input_coordinate[axis] * stride;
            stride *= input_shape[axis];
        }
        assert(output[index] == input[input_index]);
    }
}

static void test_depthwise_channel_and_spatial_shard(void)
{
    const uint32_t input_shape[4] = {1, 2, 9, 8};
    const uint32_t weight_shape[4] = {3, 1, 3, 2};
    const uint32_t output_shape[4] = {1, 3, 2, 9};
    uint16_t input[2][9][8];
    uint16_t weights[3][3][2];
    uint16_t bias[3];
    uint16_t output[3][2][9];
    float expected[3][2][9];
    for (uint32_t c = 0; c < 2; ++c)
        for (uint32_t h = 0; h < 9; ++h)
            for (uint32_t w = 0; w < 8; ++w)
                input[c][h][w] = maps_operation_f32_to_f16(
                    (float)((c * 72 + h * 8 + w) % 17));
    for (uint32_t c = 0; c < 3; ++c) {
        bias[c] = maps_operation_f32_to_f16((float)c - 1.0f);
        for (uint32_t kh = 0; kh < 3; ++kh)
            for (uint32_t kw = 0; kw < 2; ++kw)
                weights[c][kh][kw] = maps_operation_f32_to_f16(
                    (float)((c * 6 + kh * 2 + kw) % 5) - 2.0f);
    }
    assert(maps_local_depthwise_conv_f16(
               &input[0][0][0], &weights[0][0][0], bias,
               &output[0][0][0], input_shape, weight_shape, output_shape,
               1, 0, 0, 3, 1, 0, 2, 1, 2, 2, 2, 1, 2) == 0);
    for (uint32_t c = 0; c < 3; ++c) {
        const uint32_t input_channel = (3 + c) / 2 - 1;
        for (uint32_t oh = 0; oh < 2; ++oh)
            for (uint32_t ow = 0; ow < 9; ++ow) {
                float sum = maps_operation_f16_to_f32(bias[c]);
                for (uint32_t kh = 0; kh < 3; ++kh)
                    for (uint32_t kw = 0; kw < 2; ++kw) {
                        const int32_t global_h = (int32_t)((1 + oh) * 2) - 2 +
                            (int32_t)(kh * 2);
                        const int32_t global_w = (int32_t)ow - 1 +
                            (int32_t)(kw * 2);
                        if (global_h < 0 || global_h >= 9 ||
                            global_w < 0 || global_w >= 8)
                            continue;
                        sum += maps_operation_f16_to_f32(
                                   input[input_channel][global_h][global_w]) *
                            maps_operation_f16_to_f32(weights[c][kh][kw]);
                    }
                expected[c][oh][ow] = sum;
            }
    }
    for (uint32_t c = 0; c < 3; ++c)
        for (uint32_t oh = 0; oh < 2; ++oh)
            for (uint32_t ow = 0; ow < 9; ++ow)
                assert(output[c][oh][ow] ==
                       maps_operation_f32_to_f16(expected[c][oh][ow]));
}

int main(void)
{
    test_im2col_spatial_shard();
    test_rank6_transpose_nonzero_origin();
    test_depthwise_channel_and_spatial_shard();
    puts("maps operation indexing: PASS");
    return 0;
}
