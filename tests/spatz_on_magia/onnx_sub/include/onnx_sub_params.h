#ifndef ONNX_SUB_PARAMS_H_
#define ONNX_SUB_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t addr_input_a;
    uintptr_t addr_input_b;
    uintptr_t addr_res;
    uintptr_t addr_exp;
    uint32_t len;
} onnx_sub_params_t;

#endif  /* ONNX_SUB_PARAMS_H_ */
