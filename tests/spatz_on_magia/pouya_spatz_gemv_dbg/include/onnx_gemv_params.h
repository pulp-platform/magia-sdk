#ifndef ONNX_GEMV_PARAMS_H_
#define ONNX_GEMV_PARAMS_H_

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    uintptr_t addr_alpha; /* Scalar multiplier for the product A * x        */
    uintptr_t addr_beta;  /* Scalar multiplier for input vector C           */
    uintptr_t addr_A;     /* Input Matrix A (M x K)                         */
    uintptr_t addr_x;     /* Input Vector x (K)                             */
    uintptr_t addr_C;     /* Input Vector C (M) - scaled by beta            */
    uintptr_t addr_Y;     /* Output Vector Y - computed result (M)          */
    uintptr_t addr_G;     /* Output Vector G - golden model (M)             */
    bool transA;          /* Whether A should be treated as transposed      */
    uint32_t M;            /* Rows of A, length of C / Y / G                 */
    uint32_t K;            /* Columns of A, length of x                      */
} onnx_gemv_params_t;

#endif /* ONNX_GEMV_PARAMS_H_ */