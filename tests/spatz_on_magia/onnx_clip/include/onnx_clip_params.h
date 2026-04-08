#ifndef ONNX_CLIP_PARAMS_H_
#define ONNX_CLIP_PARAMS_H_

#include <stdint.h>
typedef struct {
    uintptr_t addr_input;
    uintptr_t addr_res;
    uintptr_t addr_exp;
    uintptr_t addr_min;
    uintptr_t addr_max;
    uint32_t len;
} onnx_clip_params_t;

#endif  /* ONNX_CLIP_PARAMS_H_ */
