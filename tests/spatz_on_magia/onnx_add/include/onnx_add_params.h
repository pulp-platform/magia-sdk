#ifndef ONNX_ADD_PARAMS_H_
#define ONNX_ADD_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t addr_res;
    uintptr_t addr_exp;
    uintptr_t addr_a;
    uintptr_t addr_b;
    uint32_t len;
} onnx_add_params_t;

#endif  /* ONNX_ADD_PARAMS_H_ */
