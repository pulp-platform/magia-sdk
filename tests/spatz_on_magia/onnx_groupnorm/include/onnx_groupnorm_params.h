#ifndef ONNX_GROUPNORM_PARAMS_H_
#define ONNX_GROUPNORM_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t addr_gamma;
    uintptr_t addr_beta;
    uintptr_t addr_src;
    uintptr_t addr_res;
    uintptr_t addr_exp;
    uintptr_t addr_eps;
    uint32_t num_grps;
    uint32_t len;
} onnx_groupnorm_params_t;

#endif  /* ONNX_GROUPNORM_PARAMS_H */
