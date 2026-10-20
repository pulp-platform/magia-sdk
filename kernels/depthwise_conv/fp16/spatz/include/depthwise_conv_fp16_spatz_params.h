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
    /* MAPS fused grouped im2col + GEMM. Zero selects the direct fallback. */
    uintptr_t im2col;
    uint32_t im2col_columns, im2col_stride;
    uintptr_t vector_profile; /* Optional four phase counters in shared L1. */
    uint32_t vector_direct; /* Experimental aligned-word input gathers. */
} depthwise_conv_fp16_spatz_params_t;
#endif
