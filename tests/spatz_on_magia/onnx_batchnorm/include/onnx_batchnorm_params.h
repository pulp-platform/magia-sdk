#ifndef ONNX_BATCHNORM_PARAMS_H_
#define ONNX_BATCHNORM_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t addr_gamma;
    uintptr_t addr_beta;
    uintptr_t addr_mean;
    uintptr_t addr_var;
    uintptr_t addr_src;
    uintptr_t addr_res;
    uintptr_t addr_exp;
    uintptr_t addr_eps;
    uint32_t len;
} onnx_batchnorm_params_t;

#endif  /* ONNX_BATCHNORM_PARAMS_H */
