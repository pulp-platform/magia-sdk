#include "tile.h"
#include "conv2dgemm_fp16_spatz_params.h"

/* Uncomment to accumulate the GEMM in fp32 (widening MAC + narrow to fp16) instead of fp16. */
// #define FP32_ACCUM

static void gemm(const _Float16 *A, const _Float16 *B, const _Float16 *C, _Float16 *Y, _Float16 alpha, _Float16 beta, const size_t dim_M, const size_t dim_N, const size_t dim_K)
{
    register _Float16 ZERO asm ("fs0") = 0.0f;
    const _Float16 *A_col1_elem1;
    const _Float16 *A_col1_elem2;
    const _Float16 *A_col2_elem1;
    const _Float16 *A_col2_elem2;
    const _Float16 *B_row1;
    const _Float16 *B_row2;
    const _Float16 *C_row1;
    const _Float16 *C_row2;
    _Float16 *Y_row1;
    _Float16 *Y_row2;
    size_t avl;
    size_t vl;

    /* Outer Product */
    for (int n = 0; n < dim_N; n += vl) {
        avl = dim_N - n;
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        /* 2) Unroll over two A-col at time (two B-rows at time) */
        for (int m = 0; m < (dim_M - 1); m += 2) {
            C_row1 = C + (m * dim_N + n);
            Y_row1 = Y + (m * dim_N + n);
            C_row2 = C + ((m + 1) * dim_N + n);
            Y_row2 = Y + ((m + 1) * dim_N + n);

            asm volatile ("vfmv.v.f v0, %0" :: "f"(ZERO));
            asm volatile ("vfmv.v.f v8, %0" :: "f"(ZERO));

            if (alpha != 0.0f) {
                /* 1) Unroll over two elements of each A-col at time (same B-row) */
                for (int k = 0; k < (int) dim_K - 1; k += 2) {
                    B_row1 = B + (k * dim_N + n);
                    B_row2 = B + ((k + 1) * dim_N + n);
                    A_col1_elem1 = A + (m * dim_K + k);
                    A_col2_elem1 = A + (m * dim_K + (k + 1));
                    A_col1_elem2 = A + ((m + 1) * dim_K + k);
                    A_col2_elem2 = A + ((m + 1) * dim_K + (k + 1));

                    asm volatile ("vle16.v v16, (%0)" :: "r"(B_row1));
                    asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_col1_elem1));
                    asm volatile ("vfmacc.vf v8, %0, v16" :: "f"(*A_col1_elem2));

                    asm volatile ("vle16.v v24, (%0)" :: "r"(B_row2));
                    asm volatile ("vfmacc.vf v0, %0, v24" :: "f"(*A_col2_elem1));
                    asm volatile ("vfmacc.vf v8, %0, v24" :: "f"(*A_col2_elem2));
                }

                /* Leftovers for elements of each A-col (same B-rows) */
                if (dim_K % 2) {
                    B_row1 = B + ((dim_K - 1) * dim_N + n);
                    A_col1_elem1 = A + (m * dim_K + (dim_K - 1));
                    A_col1_elem2 = A + ((m + 1) * dim_K + (dim_K - 1));

                    asm volatile ("vle16.v v16, (%0)" :: "r"(B_row1));
                    asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_col1_elem1));
                    asm volatile ("vfmacc.vf v8, %0, v16" :: "f"(*A_col1_elem2));
                }

                /* acc = alpha * A @ B */
                asm volatile ("vfmul.vf v0, v0, %0" :: "f"(alpha));
                asm volatile ("vfmul.vf v8, v8, %0" :: "f"(alpha));
            }

            /* acc += beta * C */
            if (beta != 0.0f) {
                asm volatile ("vle16.v v16, (%0)" :: "r"(C_row1));
                asm volatile ("vle16.v v24, (%0)" :: "r"(C_row2));
                asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(beta));
                asm volatile ("vfmacc.vf v8, %0, v24" :: "f"(beta));
            }

            asm volatile ("vse16.v v0, (%0)" :: "r"(Y_row1));
            asm volatile ("vse16.v v8, (%0)" :: "r"(Y_row2));
        }

        /* Leftovers for A-cols and B-rows  */
        if (dim_M % 2) {
            C_row1 = C + ((dim_M - 1) * dim_N + n);
            Y_row1 = Y + ((dim_M - 1) * dim_N + n);

            asm volatile ("vfmv.v.f v0, %0" :: "f"(ZERO));

            if (alpha != 0.0f) {
                for (int k = 0; k < (int) dim_K - 1; k += 2) {
                    B_row1 = B + (k * dim_N + n);
                    B_row2 = B + ((k + 1) * dim_N + n);
                    A_col1_elem1 = A + ((dim_M - 1) * dim_K + k);
                    A_col2_elem1 = A + ((dim_M - 1) * dim_K + (k + 1));


                    asm volatile ("vle16.v v16, (%0)" :: "r"(B_row1));
                    asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_col1_elem1));

                    asm volatile ("vle16.v v24, (%0)" :: "r"(B_row2));
                    asm volatile ("vfmacc.vf v0, %0, v24" :: "f"(*A_col2_elem1));
                }

                if (dim_K % 2) {
                    B_row1 = B + ((dim_K - 1) * dim_N + n);
                    A_col1_elem1 = A + ((dim_M - 1) * dim_K + (dim_K - 1));

                    asm volatile ("vle16.v v16, (%0)" :: "r"(B_row1));
                    asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_col1_elem1));
                }

                /* acc = alpha * A @ B */
                asm volatile ("vfmul.vf v0, v0, %0" :: "f"(alpha));
            }

            /* acc += beta * C */
            if (beta != 0.0f) {
                asm volatile ("vle16.v v16, (%0)" :: "r"(C_row1));
                asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(beta));
            }

            asm volatile ("vse16.v v0, (%0)" :: "r"(Y_row1));
        }
    }
}

static void gemm_grouped(const _Float16 *A, const _Float16 *B, const _Float16 *C, _Float16 *Y, _Float16 alpha, _Float16 beta, const size_t dim_M, const size_t dim_N, const size_t dim_Kg, const size_t cout_g, const size_t oc_start)
{
    register _Float16 ZERO asm ("fs0") = 0.0f;
    const _Float16 *A_row;
    const _Float16 *B_grp;
    const _Float16 *B_row;
    const _Float16 *C_row;
    _Float16 *Y_row;
    size_t avl;
    size_t vl;

    /* Grouped/depthwise: each output channel's compact weights (dim_Kg) multiply only its own
       group's im2col block (rows [g * dim_Kg, (g + 1) * dim_Kg) of B), avoiding the full-K dense
       product with a block-diagonal weight matrix. One dense GEMV per output row. */
    for (int n = 0; n < dim_N; n += vl) {
        avl = dim_N - n;
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        for (int m = 0; m < dim_M; m++) {
            int g = (oc_start + m) / cout_g;
            A_row = A + m * dim_Kg;
            B_grp = B + g * dim_Kg * dim_N;
            C_row = C + (m * dim_N + n);
            Y_row = Y + (m * dim_N + n);

            asm volatile ("vfmv.v.f v0, %0" :: "f"(ZERO));

            if (alpha != 0.0f) {
                for (int k = 0; k < (int) dim_Kg; k++) {
                    B_row = B_grp + (k * dim_N + n);
                    asm volatile ("vle16.v v16, (%0)" :: "r"(B_row));
                    asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(A_row[k]));
                }

                asm volatile ("vfmul.vf v0, v0, %0" :: "f"(alpha));
            }

            if (beta != 0.0f) {
                asm volatile ("vle16.v v16, (%0)" :: "r"(C_row));
                asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(beta));
            }

            asm volatile ("vse16.v v0, (%0)" :: "r"(Y_row));
        }
    }
}

#ifdef FP32_ACCUM

static void gemm_accfp32(const _Float16 *A, const _Float16 *B, const _Float16 *C, _Float16 *Y, _Float16 alpha, _Float16 beta, const size_t dim_M, const size_t dim_N, const size_t dim_K)
{
    register _Float16 ZERO asm ("fs0") = 0.0f;
    const _Float16 *B_row;
    const _Float16 *C_row;
    _Float16 *Y_row;
    size_t avl;
    size_t vl;

    /* Each output row is a dot product over dim_K accumulated in fp32: v8 (e32) is built with a
       widening multiply on the first term and widening MACs on the rest, then narrowed to fp16. */
    for (int n = 0; n < dim_N; n += vl) {
        avl = dim_N - n;
        asm volatile ("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl));

        for (int m = 0; m < dim_M; m++) {
            C_row = C + (m * dim_N + n);
            Y_row = Y + (m * dim_N + n);

            if (alpha != 0.0f) {
                B_row = B + n;
                asm volatile ("vle16.v v0, (%0)" :: "r"(B_row));
                asm volatile ("vfwmul.vf v8, v0, %0" :: "f"(*(A + m * dim_K)));

                for (int k = 1; k < (int) dim_K; k++) {
                    B_row = B + (k * dim_N + n);
                    asm volatile ("vle16.v v0, (%0)" :: "r"(B_row));
                    asm volatile ("vfwmacc.vf v8, %0, v0" :: "f"(*(A + m * dim_K + k)));
                }

                asm volatile ("vfncvt.f.f.w v16, v8");
                asm volatile ("vfmul.vf v16, v16, %0" :: "f"(alpha));
            } else {
                asm volatile ("vfmv.v.f v16, %0" :: "f"(ZERO));
            }

            if (beta != 0.0f) {
                asm volatile ("vle16.v v0, (%0)" :: "r"(C_row));
                asm volatile ("vfmacc.vf v16, %0, v0" :: "f"(beta));
            }

            asm volatile ("vse16.v v16, (%0)" :: "r"(Y_row));
        }
    }
}

static void gemm_grouped_accfp32(const _Float16 *A, const _Float16 *B, const _Float16 *C, _Float16 *Y, _Float16 alpha, _Float16 beta, const size_t dim_M, const size_t dim_N, const size_t dim_Kg, const size_t cout_g, const size_t oc_start)
{
    register _Float16 ZERO asm ("fs0") = 0.0f;
    const _Float16 *A_row;
    const _Float16 *B_grp;
    const _Float16 *B_row;
    const _Float16 *C_row;
    _Float16 *Y_row;
    size_t avl;
    size_t vl;

    for (int n = 0; n < dim_N; n += vl) {
        avl = dim_N - n;
        asm volatile ("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl));

        for (int m = 0; m < dim_M; m++) {
            int g = (oc_start + m) / cout_g;
            A_row = A + m * dim_Kg;
            B_grp = B + g * dim_Kg * dim_N;
            C_row = C + (m * dim_N + n);
            Y_row = Y + (m * dim_N + n);

            if (alpha != 0.0f) {
                B_row = B_grp + n;
                asm volatile ("vle16.v v0, (%0)" :: "r"(B_row));
                asm volatile ("vfwmul.vf v8, v0, %0" :: "f"(A_row[0]));

                for (int k = 1; k < (int) dim_Kg; k++) {
                    B_row = B_grp + (k * dim_N + n);
                    asm volatile ("vle16.v v0, (%0)" :: "r"(B_row));
                    asm volatile ("vfwmacc.vf v8, %0, v0" :: "f"(A_row[k]));
                }

                asm volatile ("vfncvt.f.f.w v16, v8");
                asm volatile ("vfmul.vf v16, v16, %0" :: "f"(alpha));
            } else {
                asm volatile ("vfmv.v.f v16, %0" :: "f"(ZERO));
            }

            if (beta != 0.0f) {
                asm volatile ("vle16.v v0, (%0)" :: "r"(C_row));
                asm volatile ("vfmacc.vf v16, %0, v0" :: "f"(beta));
            }

            asm volatile ("vse16.v v16, (%0)" :: "r"(Y_row));
        }
    }
}

#endif

int conv2dgemm_fp16_spatz_task(void)
{
    volatile conv2dgemm_fp16_spatz_params_t *params;
    uintptr_t params_addr;
    _Float16 alpha;
    _Float16 beta;
    _Float16 *A;
    _Float16 *B;
    _Float16 *C;
    _Float16 *Y;
    size_t M;
    size_t N;
    size_t K;
    size_t K_g;
    size_t cout_g;
    size_t oc_start;
    size_t group;
    size_t c_out;
    size_t n_batches;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile conv2dgemm_fp16_spatz_params_t *) params_addr;

    alpha = *(_Float16 *) params->alpha;
    beta = *(_Float16 *) params->beta;
    A = (_Float16 *) params->shard_A;
    B = (_Float16 *) params->shard_B;
    C = (_Float16 *) params->shard_C;
    Y = (_Float16 *) params->shard_Y;
    n_batches = params->n_batches;
    oc_start = params->oc_start;
    group = params->group;
    c_out = params->c_out;
    M = params->M;
    N = params->N;
    K = params->K;

    if (M == 0)
        return 0;

    K_g = K / group;
    cout_g = c_out / group;

    /* Weights (A) and bias (C) are shared across batches; the im2col matrix (B) and the output
       (Y) are batched, so advance them by one 2D slice per batch. group == 1 is a plain dense
       GEMM; grouped/depthwise convs apply each output channel's compact weights to only its
       group's block of B. */
    for (size_t b = 0; b < n_batches; b++) {
#ifdef FP32_ACCUM
        if (group == 1)
            gemm_accfp32(A, B + b * (K * N), C, Y + b * (M * N), alpha, beta, M, N, K_g);
        else
            gemm_grouped_accfp32(A, B + b * (K * N), C, Y + b * (M * N), alpha, beta, M, N, K_g, cout_g, oc_start);
#else
        if (group == 1)
            gemm(A, B + b * (K * N), C, Y + b * (M * N), alpha, beta, M, N, K_g);
        else
            gemm_grouped(A, B + b * (K * N), C, Y + b * (M * N), alpha, beta, M, N, K_g, cout_g, oc_start);
#endif
    }

    return 0;
}
