#ifndef TRANSPOSE_STRIDED_FP16_SPATZ_PARAMS_H
#define TRANSPOSE_STRIDED_FP16_SPATZ_PARAMS_H
#include <stdint.h>
typedef struct {
    uintptr_t input, output;
    uint32_t rank, shape[6], source_stride[6], output_stride[6];
    uint32_t packed_pairs;
} transpose_strided_fp16_spatz_params_t;
#endif
