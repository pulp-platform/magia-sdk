#include "tile.h"
#include "onnx_gemv_params.h"

/*
 * GEMV: Y = alpha * A * x + beta * C
 *
 * Unlike GEMM, there is no free N dimension to vectorize over, so instead
 * we vectorize across M (the output/row dimension): each vector lane
 * accumulates one output element Y[m]. For a fixed k, we broadcast the
 * scalar x[k] and multiply-accumulate it against a vector of A[m..m+vl][k]
 * (one element per lane) - the same "outer product" trick used in the GEMM
 * kernel, just with A and x swapping roles relative to A and B there.
 *
 * Non-transposed A is (M x K) row-major, so A[m][k] for fixed k / varying m
 * is strided (stride = K * sizeof(_Float16)) -> vlse16.v.
 * Transposed A is (K x M) row-major, so A[k][m] for fixed k / varying m is
 * contiguous -> vle16.v.
 */

static void gemv(const _Float16 *A,
                 const _Float16 *x,
                 const _Float16 *C,
                 _Float16 *Y,
                 _Float16 alpha,
                 _Float16 beta,
                 const size_t dim_M,
                 const size_t dim_K)
{
    register _Float16 ZERO asm("fs0") = 0.0f;
    const _Float16 *A_col;
    const _Float16 *C_seg;
    _Float16 *Y_seg;
    size_t stride;
    size_t avl;
    size_t vl;

    stride = dim_K * sizeof(_Float16);

    for (unsigned int m = 0; m < dim_M; m += vl) {
        avl = dim_M - m;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        C_seg = C + m;
        Y_seg = Y + m;

        asm volatile("vfmv.v.f v0, %0" ::"f"(ZERO));

        if (alpha != 0.0f) {
            for (unsigned int k = 0; k < dim_K; k++) {
                A_col = A + (m * dim_K + k);
                asm volatile("vlse16.v v16, (%0), %1" ::"r"(A_col), "r"(stride));
                asm volatile("vfmacc.vf v0, %0, v16" ::"f"(*(x + k)));
            }

            /* acc = alpha * (A @ x) */
            asm volatile("vfmul.vf v0, v0, %0" ::"f"(alpha));
        }

        /* acc += beta * C */
        if (beta != 0.0f) {
            asm volatile("vle16.v v16, (%0)" ::"r"(C_seg));
            asm volatile("vfmacc.vf v0, %0, v16" ::"f"(beta));
        }

        asm volatile("vse16.v v0, (%0)" ::"r"(Y_seg));
    }
}

static void gemv_Atrans(const _Float16 *A,
                        const _Float16 *x,
                        const _Float16 *C,
                        _Float16 *Y,
                        _Float16 alpha,
                        _Float16 beta,
                        const size_t dim_M,
                        const size_t dim_K)
{
    register _Float16 ZERO asm("fs0") = 0.0f;
    const _Float16 *A_row;
    const _Float16 *C_seg;
    _Float16 *Y_seg;
    size_t avl;
    size_t vl;

    for (unsigned int m = 0; m < dim_M; m += vl) {
        avl = dim_M - m;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        C_seg = C + m;
        Y_seg = Y + m;

        asm volatile("vfmv.v.f v0, %0" ::"f"(ZERO));

        if (alpha != 0.0f) {
            for (unsigned int k = 0; k < dim_K; k++) {
                A_row = A + (k * dim_M + m);
                asm volatile("vle16.v v16, (%0)" ::"r"(A_row));
                asm volatile("vfmacc.vf v0, %0, v16" ::"f"(*(x + k)));
            }

            /* acc = alpha * (A^T @ x) */
            asm volatile("vfmul.vf v0, v0, %0" ::"f"(alpha));
        }

        /* acc += beta * C */
        if (beta != 0.0f) {
            asm volatile("vle16.v v16, (%0)" ::"r"(C_seg));
            asm volatile("vfmacc.vf v0, %0, v16" ::"f"(beta));
        }

        asm volatile("vse16.v v0, (%0)" ::"r"(Y_seg));
    }
}

int onnx_gemv_task(void)
{
    volatile onnx_gemv_params_t *params;
    uintptr_t params_addr;
    _Float16 alpha;
    _Float16 beta;
    _Float16 *A;
    _Float16 *x;
    _Float16 *C;
    _Float16 *Y;
    bool transA;
    size_t M;
    size_t K;

    params_addr = mmio32(SPATZ_DATA);
    params      = (volatile onnx_gemv_params_t *)params_addr;

    alpha  = *(_Float16 *)params->addr_alpha;
    beta   = *(_Float16 *)params->addr_beta;
    A      = (_Float16 *)params->addr_A;
    x      = (_Float16 *)params->addr_x;
    C      = (_Float16 *)params->addr_C;
    Y      = (_Float16 *)params->addr_Y;
    transA = params->transA;
    M      = params->M;
    K      = params->K;

    if (transA)
        gemv_Atrans(A, x, C, Y, alpha, beta, M, K);
    else
        gemv(A, x, C, Y, alpha, beta, M, K);

    return 0;
}