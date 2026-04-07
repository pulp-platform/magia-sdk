#ifndef ONNX_HARDSIGMOID_PARAMS_H_
#define ONNX_HARDSIGMOID_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t addr_alpha;
    uintptr_t addr_beta;
    uintptr_t addr_input;
    uintptr_t addr_res;
    uintptr_t addr_exp;
    uint32_t len;
} onnx_hardsigmoid_params_t;

#endif  /* ONNX_HARDSIGMOID_PARAMS_H_ */
