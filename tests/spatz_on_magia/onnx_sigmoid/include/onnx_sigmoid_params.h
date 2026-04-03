#ifndef ONNX_SIGMOID_PARAMS_H_
#define ONNX_SIGMOID_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t addr_input;
    uintptr_t addr_res;
    uintptr_t addr_exp;
    uint32_t len;
} onnx_sigmoid_params_t;

#endif  /* ONNX_SIGMOID_PARAMS_H_ */
