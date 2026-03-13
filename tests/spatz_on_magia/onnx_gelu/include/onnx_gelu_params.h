#ifndef ONNX_GELU_PARAMS_H_
#define ONNX_GELU_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t addr_res;
    uintptr_t addr_exp;
    uintptr_t addr_src;
    uint32_t len;
} onnx_gelu_params_t;

#endif  /* ONNX_GELU_PARAMS_H_ */
