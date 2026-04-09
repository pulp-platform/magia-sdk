#ifndef ONNX_GEMM_PARAMS_H_
#define ONNX_GEMM_PARAMS_H_

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    uintptr_t addr_alpha;       /* Scalar multiplier for the product of input tensors A * B */
    uintptr_t addr_beta;        /* Scalar multiplier for input tensor C                     */
    uintptr_t addr_A;           /* Input Tensor A                                           */
    uintptr_t addr_B;           /* Input Tensor B                                           */
    uintptr_t addr_C;           /* Input Tensor C                                           */
    uintptr_t addr_Y;           /* Output Tensor Y - computed result                        */
    uintptr_t addr_G;           /* Output Tensor G - golden model                           */
    bool transA;                /* Whether A should be transposed                           */
    bool transB;                /* Whether A should be transposed                           */
    uint32_t M;                 /* Rows of A anc C                                          */
    uint32_t N;                 /* Columns of B and C                                       */
    uint32_t K;                 /* Columns of A - Rows of B                                 */
} onnx_gemm_params_t;

#endif  /* ONNX_GEMM_PARAMS_H_ */
