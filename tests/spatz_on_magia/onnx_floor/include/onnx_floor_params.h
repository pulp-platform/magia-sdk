#ifndef ONNX_FLOOR_PARAMS_H_
#define ONNX_FLOOR_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t addr_input;
    uintptr_t addr_res;
    uintptr_t addr_exp;
    uint32_t len;
} onnx_floor_params_t;

#endif  /* ONNX_FLOOR_PARAMS_H_ */
