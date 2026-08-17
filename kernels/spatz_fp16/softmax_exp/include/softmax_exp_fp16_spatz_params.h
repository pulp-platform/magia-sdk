#ifndef SOFTMAX_EXP_FP16_SPATZ_PARAMS_H_
#define SOFTMAX_EXP_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t input;
    uintptr_t output;
    uint32_t len;
} softmax_exp_fp16_spatz_params_t;

#endif
