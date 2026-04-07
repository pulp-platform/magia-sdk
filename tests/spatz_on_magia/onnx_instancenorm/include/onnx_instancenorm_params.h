#ifndef ONNX_INSTANCENORM_PARAMS_H_
#define ONNX_INSTANCENORM_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t addr_gamma;
    uintptr_t addr_beta;
    uintptr_t addr_src;
    uintptr_t addr_res;
    uintptr_t addr_exp;
    uintptr_t addr_eps;
    uint32_t len;
} onnx_instancenorm_params_t;

#endif  /* ONNX_INSTANCENORM_PARAMS_H */
