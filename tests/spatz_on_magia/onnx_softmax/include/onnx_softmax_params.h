#ifndef ONNX_SOFTMAX_PARAMS_H_
#define ONNX_SOFTMAX_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t addr_res;
    uintptr_t addr_exp;
    uintptr_t addr_src;
    uint32_t len;
} onnx_softmax_params_t;

#endif  /* ONNX_SOFTMAX_PARAMS_H */
