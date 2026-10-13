#ifndef DEPTHWISE_CONV_FP16_SPATZ_PARAMS_H
#define DEPTHWISE_CONV_FP16_SPATZ_PARAMS_H
#include <stdint.h>
typedef struct {
    uintptr_t input, weight, bias, output;
    uint32_t input_shape[4], weight_shape[4], output_shape[4];
    uint32_t input_strides[4], weight_strides[4], output_strides[4];
    uint32_t input_origin_c, input_origin_h, input_origin_w;
    uint32_t weight_origin_c, output_origin_h, output_origin_w;
    uint32_t stride_h, stride_w, dilation_h, dilation_w;
    uint32_t pad_top, pad_left, channel_multiplier, bias_stride;
} depthwise_conv_fp16_spatz_params_t;
#endif
