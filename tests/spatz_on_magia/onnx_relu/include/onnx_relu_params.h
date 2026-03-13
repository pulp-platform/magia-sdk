#ifndef ONNX_RELU_PARAMS_H_
#define ONNX_RELU_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t addr_res;
    uintptr_t addr_exp;
    uintptr_t addr_src;
    uint32_t len;
} onnx_relu_params_t;

#endif  /* ONNX_RELU_PARAMS_H */
