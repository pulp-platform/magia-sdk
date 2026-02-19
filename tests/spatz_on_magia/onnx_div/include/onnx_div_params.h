#ifndef ONNX_DIV_PARAMS_H_
#define ONNX_DIV_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t addr_res;
    uintptr_t addr_exp;
    uintptr_t addr_a;
    uintptr_t addr_b;
    uint32_t len;
} onnx_div_params_t;

#endif  /* ONNX_DIV_PARAMS_H_ */
