#include "tile.h"
#include "onnx_gemm_params.h"

static inline uint16_t get_raw(const _Float16 val)
{
    uint16_t raw;
    memcpy(&raw, &val, sizeof(raw));
    return raw;
}

static inline void print_vector_raw(const _Float16 *vec, size_t len)
{
    for (size_t i = 0; i < len; i++) {
        printf("%d) %x\n", i, get_raw(vec[i]));
    }
}

static inline void print_matrix_raw(const _Float16 *mat, size_t rows, size_t cols)
{
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            size_t i = r * cols + c;
            printf("%x ", get_raw(mat[i]));
        }
        printf("\n");
    }
}

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
                for (int k = 0; k < dim_K; k += 2) {
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
                for (int k = 0; k < dim_K; k += 2) {
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
                    asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_col1_elem2));
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

static void gemm_Atrans(const _Float16 *A, const _Float16 *B, const _Float16 *C, _Float16 *Y, _Float16 alpha, _Float16 beta, const size_t dim_M, const size_t dim_N, const size_t dim_K)
{
    register _Float16 ZERO asm ("fs0") = 0.0f;
    const _Float16 *A_row1_elem1;
    const _Float16 *A_row1_elem2;
    const _Float16 *A_row2_elem1;
    const _Float16 *A_row2_elem2;
    const _Float16 *B_row1;
    const _Float16 *B_row2;
    const _Float16 *C_row1;
    const _Float16 *C_row2;
    _Float16 *Y_row1;
    _Float16 *Y_row2;
    size_t avl;
    size_t vl;

    for (int n = 0; n < dim_N; n += vl) {
        avl = dim_N - n;
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        for (int m = 0; m < (dim_M - 1); m += 2) {
            C_row1 = C + (m * dim_N + n);
            Y_row1 = Y + (m * dim_N + n);
            C_row2 = C + ((m + 1) * dim_N + n);
            Y_row2 = Y + ((m + 1) * dim_N + n);

            asm volatile ("vfmv.v.f v0, %0" :: "f"(ZERO));
            asm volatile ("vfmv.v.f v8, %0" :: "f"(ZERO));

            if (alpha != 0.0f) {
                for (int k = 0; k < dim_K; k += 2) {
                    B_row1 = B + (k * dim_N + n);
                    B_row2 = B + ((k + 1) * dim_N + n);
                    A_row1_elem1 = A + (k * dim_M + m);
                    A_row1_elem2 = A + (k * dim_M + (m + 1));
                    A_row2_elem1 = A + ((k + 1) * dim_M + m);
                    A_row2_elem2 = A + ((k + 1) * dim_M + (m + 1));

                    asm volatile ("vle16.v v16, (%0)" :: "r"(B_row1));
                    asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_row1_elem1));
                    asm volatile ("vfmacc.vf v8, %0, v16" :: "f"(*A_row1_elem2));

                    asm volatile ("vle16.v v24, (%0)" :: "r"(B_row2));
                    asm volatile ("vfmacc.vf v0, %0, v24" :: "f"(*A_row2_elem1));
                    asm volatile ("vfmacc.vf v8, %0, v24" :: "f"(*A_row2_elem2));
                }

                if (dim_K % 2) {
                    B_row1 = B + ((dim_K - 1) * dim_N + n);
                    A_row1_elem1 = A + ((dim_K - 1) * dim_M + m);
                    A_row1_elem2 = A + ((dim_K - 1) * dim_M + (m + 1));

                    asm volatile ("vle16.v v16, (%0)" :: "r"(B_row1));
                    asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_row1_elem1));
                    asm volatile ("vfmacc.vf v8, %0, v16" :: "f"(*A_row1_elem2));
                }

                asm volatile ("vfmul.vf v0, v0, %0" :: "f"(alpha));
                asm volatile ("vfmul.vf v8, v8, %0" :: "f"(alpha));
            }

            if (beta != 0.0f) {
                asm volatile ("vle16.v v16, (%0)" :: "r"(C_row1));
                asm volatile ("vle16.v v24, (%0)" :: "r"(C_row2));
                asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(beta));
                asm volatile ("vfmacc.vf v8, %0, v24" :: "f"(beta));
            }

            asm volatile ("vse16.v v0, (%0)" :: "r"(Y_row1));
            asm volatile ("vse16.v v8, (%0)" :: "r"(Y_row2));
        }

        if (dim_M % 2) {
            C_row1 = C + ((dim_M - 1) * dim_N + n);
            Y_row1 = Y + ((dim_M - 1) * dim_N + n);

            asm volatile ("vfmv.v.f v0, %0" :: "f"(ZERO));
            asm volatile ("vfmv.v.f v8, %0" :: "f"(ZERO));

            if (alpha != 0.0f) {
                for (int k = 0; k < dim_K; k += 2) {
                    B_row1 = B + (k * dim_N + n);
                    B_row2 = B + ((k + 1) * dim_N + n);
                    A_row1_elem1 = A + (k * dim_M + (dim_M - 1));
                    A_row2_elem1 = A + ((k + 1) * dim_M + (dim_M - 1));

                    asm volatile ("vle16.v v16, (%0)" :: "r"(B_row1));
                    asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_row1_elem1));

                    asm volatile ("vle16.v v24, (%0)" :: "r"(B_row2));
                    asm volatile ("vfmacc.vf v0, %0, v24" :: "f"(*A_row2_elem1));
                }

                if (dim_K % 2) {
                    B_row1 = B + ((dim_K - 1) * dim_N + n);
                    A_row1_elem1 = A + ((dim_K - 1) * dim_M + (dim_M - 1));

                    asm volatile ("vle16.v v16, (%0)" :: "r"(B_row1));
                    asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_row1_elem1));
                }

                asm volatile ("vfmul.vf v0, v0, %0" :: "f"(alpha));
            }

            if (beta != 0.0f) {
                asm volatile ("vle16.v v16, (%0)" :: "r"(C_row1));
                asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(beta));
            }

            asm volatile ("vse16.v v0, (%0)" :: "r"(Y_row1));
        }
    }
}
static void gemm_Btrans(const _Float16 *A, const _Float16 *B, const _Float16 *C, _Float16 *Y, _Float16 alpha, _Float16 beta, const size_t dim_M, const size_t dim_N, const size_t dim_K)
{
    register _Float16 ZERO asm ("fs0") = 0.0f;
    const _Float16 *A_col1_elem1;
    const _Float16 *A_col1_elem2;
    const _Float16 *A_col2_elem1;
    const _Float16 *A_col2_elem2;
    const _Float16 *B_col1;
    const _Float16 *B_col2;
    const _Float16 *C_row1;
    const _Float16 *C_row2;
    _Float16 *Y_row1;
    _Float16 *Y_row2;
    size_t stride;
    size_t avl;
    size_t vl;

    stride = dim_K * sizeof(_Float16);

    for (int n = 0; n < dim_N; n += vl) {
        avl = dim_N - n;
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        for (int m = 0; m < (dim_M - 1); m += 2) {
            C_row1 = C + (m * dim_N + n);
            Y_row1 = Y + (m * dim_N + n);
            C_row2 = C + ((m + 1) * dim_N + n);
            Y_row2 = Y + ((m + 1) * dim_N + n);

            asm volatile ("vfmv.v.f v0, %0" :: "f"(ZERO));
            asm volatile ("vfmv.v.f v8, %0" :: "f"(ZERO));

            if (alpha != 0.0f) {
                for (int k = 0; k < dim_K; k += 2) {
                    B_col1 = B + (n * dim_K + k);
                    B_col2 = B + (n * dim_K + (k + 1));
                    A_col1_elem1 = A + (m * dim_K + k);
                    A_col2_elem1 = A + (m * dim_K + (k + 1));
                    A_col1_elem2 = A + ((m + 1) * dim_K + k);
                    A_col2_elem2 = A + ((m + 1) * dim_K + (k + 1));

                    asm volatile ("vlse16.v v16, (%0), %1" :: "r"(B_col1), "r"(stride));
                    asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_col1_elem1));
                    asm volatile ("vfmacc.vf v8, %0, v16" :: "f"(*A_col1_elem2));

                    asm volatile ("vlse16.v v24, (%0), %1" :: "r"(B_col2), "r"(stride));
                    asm volatile ("vfmacc.vf v0, %0, v24" :: "f"(*A_col2_elem1));
                    asm volatile ("vfmacc.vf v8, %0, v24" :: "f"(*A_col2_elem2));
                }

                if (dim_K % 2) {
                    B_col1 = B + (n * dim_K + (dim_K - 1));
                    A_col1_elem1 = A + (m * dim_K + (dim_K - 1));
                    A_col1_elem2 = A + ((m + 1) * dim_K + (dim_K - 1));

                    asm volatile ("vlse16.v v16, (%0), %1" :: "r"(B_col1), "r"(stride));
                    asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_col1_elem1));
                    asm volatile ("vfmacc.vf v8, %0, v16" :: "f"(*A_col1_elem2));
                }

                asm volatile ("vfmul.vf v0, v0, %0" :: "f"(alpha));
                asm volatile ("vfmul.vf v8, v8, %0" :: "f"(alpha));
            }

            if (beta != 0.0f) {
                asm volatile ("vle16.v v16, (%0)" :: "r"(C_row1));
                asm volatile ("vle16.v v24, (%0)" :: "r"(C_row2));
                asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(beta));
                asm volatile ("vfmacc.vf v8, %0, v24" :: "f"(beta));
            }

            asm volatile ("vse16.v v0, (%0)" :: "r"(Y_row1));
            asm volatile ("vse16.v v8, (%0)" :: "r"(Y_row2));
        }

        if (dim_M % 2) {
            C_row1 = C + ((dim_M - 1) * dim_N + n);
            Y_row1 = Y + ((dim_M - 1) * dim_N + n);

            asm volatile ("vfmv.v.f v0, %0" :: "f"(ZERO));

            if (alpha != 0.0f) {
                for (int k = 0; k < dim_K; k += 2) {
                    B_col1 = B + (n * dim_K + k);
                    B_col2 = B + (n * dim_K + (k + 1));
                    A_col1_elem1 = A + ((dim_M - 1) * dim_K + k);
                    A_col2_elem1 = A + ((dim_M - 1) * dim_K + (k + 1));

                    asm volatile ("vlse16.v v16, (%0), %1" :: "r"(B_col1), "r"(stride));
                    asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_col1_elem1));

                    asm volatile ("vlse16.v v24, (%0), %1" :: "r"(B_col2), "r"(stride));
                    asm volatile ("vfmacc.vf v0, %0, v24" :: "f"(*A_col2_elem1));
                }

                if (dim_K % 2) {
                    B_col1 = B + (n * dim_K + (dim_K - 1));
                    A_col1_elem1 = A + ((dim_M - 1) * dim_K + (dim_K - 1));

                    asm volatile ("vlse16.v v16, (%0), %1" :: "r"(B_col1), "r"(stride));
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

static void gemm_ABtrans(const _Float16 *A, const _Float16 *B, const _Float16 *C, _Float16 *Y, _Float16 alpha, _Float16 beta, const size_t dim_M, const size_t dim_N, const size_t dim_K)
{
    register _Float16 ZERO asm ("fs0") = 0.0f;
    const _Float16 *A_row1_elem1;
    const _Float16 *A_row1_elem2;
    const _Float16 *A_row2_elem1;
    const _Float16 *A_row2_elem2;
    const _Float16 *B_col1;
    const _Float16 *B_col2;
    const _Float16 *C_row1;
    const _Float16 *C_row2;
    _Float16 *Y_row1;
    _Float16 *Y_row2;
    size_t stride;
    size_t avl;
    size_t vl;

    stride = dim_K * sizeof(_Float16);

    for (int n = 0; n < dim_N; n += vl) {
        avl = dim_N - n;
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        for (int m = 0; m < (dim_M - 1); m += 2) {
            C_row1 = C + (m * dim_N + n);
            Y_row1 = Y + (m * dim_N + n);
            C_row2 = C + ((m + 1) * dim_N + n);
            Y_row2 = Y + ((m + 1) * dim_N + n);

            asm volatile ("vfmv.v.f v0, %0" :: "f"(ZERO));
            asm volatile ("vfmv.v.f v8, %0" :: "f"(ZERO));

            if (alpha != 0.0f) {
                for (int k = 0; k < dim_K; k += 2) {
                    B_col1 = B + (n * dim_K + k);
                    B_col2 = B + (n * dim_K + (k + 1));
                    A_row1_elem1 = A + (k * dim_M + m);
                    A_row1_elem2 = A + (k * dim_M + (m + 1));
                    A_row2_elem1 = A + ((k + 1) * dim_M + m);
                    A_row2_elem2 = A + ((k + 1) * dim_M + (m + 1));

                    asm volatile ("vlse16.v v16, (%0), %1" :: "r"(B_col1), "r"(stride));
                    asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_row1_elem1));
                    asm volatile ("vfmacc.vf v8, %0, v16" :: "f"(*A_row1_elem2));

                    asm volatile ("vlse16.v v24, (%0), %1" :: "r"(B_col2), "r"(stride));
                    asm volatile ("vfmacc.vf v0, %0, v24" :: "f"(*A_row2_elem1));
                    asm volatile ("vfmacc.vf v8, %0, v24" :: "f"(*A_row2_elem2));
                }

                if (dim_K % 2) {
                    B_col1 = B + (n * dim_K + (dim_K - 1));
                    A_row1_elem1 = A + ((dim_K - 1) * dim_M + m);
                    A_row1_elem2 = A + ((dim_K - 1) * dim_M + (m + 1));

                    asm volatile ("vlse16.v v16, (%0), %1" :: "r"(B_col1), "r"(stride));
                    asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_row1_elem1));
                    asm volatile ("vfmacc.vf v8, %0, v16" :: "f"(*A_row1_elem2));
                }

                asm volatile ("vfmul.vf v0, v0, %0" :: "f"(alpha));
                asm volatile ("vfmul.vf v8, v8, %0" :: "f"(alpha));
            }

            if (beta != 0.0f) {
                asm volatile ("vle16.v v16, (%0)" :: "r"(C_row1));
                asm volatile ("vle16.v v24, (%0)" :: "r"(C_row2));
                asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(beta));
                asm volatile ("vfmacc.vf v8, %0, v24" :: "f"(beta));
            }

            asm volatile ("vse16.v v0, (%0)" :: "r"(Y_row1));
            asm volatile ("vse16.v v8, (%0)" :: "r"(Y_row2));
        }

        if (dim_M % 2) {
            C_row1 = C + ((dim_M - 1) * dim_N + n);
            Y_row1 = Y + ((dim_M - 1) * dim_N + n);

            asm volatile ("vfmv.v.f v0, %0" :: "f"(ZERO));

            if (alpha != 0.0f) {
                for (int k = 0; k < dim_K; k += 2) {
                    B_col1 = B + (n * dim_K + k);
                    B_col2 = B + (n * dim_K + (k + 1));
                    A_row1_elem1 = A + (k * dim_M + (dim_M - 1));
                    A_row2_elem1 = A + ((k + 1) * dim_M + (dim_M - 1));

                    asm volatile ("vlse16.v v16, (%0), %1" :: "r"(B_col1), "r"(stride));
                    asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_row1_elem1));

                    asm volatile ("vlse16.v v24, (%0), %1" :: "r"(B_col2), "r"(stride));
                    asm volatile ("vfmacc.vf v0, %0, v24" :: "f"(*A_row2_elem1));
                }

                if (dim_K % 2) {
                    B_col1 = B + (n * dim_K + (dim_K - 1));
                    A_row1_elem1 = A + ((dim_M - 1) * dim_K + (dim_K - 1));

                    asm volatile ("vlse16.v v16, (%0), %1" :: "r"(B_col1), "r"(stride));
                    asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(*A_row1_elem1));
                }

                asm volatile ("vfmul.vf v0, v0, %0" :: "f"(alpha));
            }

            if (beta != 0.0f) {
                asm volatile ("vle16.v v16, (%0)" :: "r"(C_row1));
                asm volatile ("vfmacc.vf v0, %0, v16" :: "f"(beta));
            }

            asm volatile ("vse16.v v0, (%0)" :: "r"(Y_row1));
        }
    }
}

int onnx_gemm_task(void)
{
    volatile onnx_gemm_params_t *params;
    uintptr_t params_addr;
    _Float16 alpha;
    _Float16 beta;
    _Float16 *A;
    _Float16 *B;
    _Float16 *C;
    _Float16 *Y;
    bool transA;
    bool transB;
    size_t M;
    size_t N;
    size_t K;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_gemm_params_t *) params_addr;

    alpha = *(_Float16 *) params->addr_alpha;
    beta = *(_Float16 *) params->addr_beta;
    A = (_Float16 *) params->addr_A;
    B = (_Float16 *) params->addr_B;
    C = (_Float16 *) params->addr_C;
    Y = (_Float16 *) params->addr_Y;
    transA = params->transA;
    transB = params->transB;
    M = params->M;
    N = params->N;
    K = params->K;

    if (transA && transB)
        gemm_ABtrans(A, B, C, Y, alpha, beta, M, N, K);
    else if (transA)
        gemm_Atrans(A, B, C, Y, alpha, beta, M, N, K);
    else if (transB)
        gemm_Btrans(A, B, C, Y, alpha, beta, M, N, K);
    else
        gemm(A, B, C, Y, alpha, beta, M, N, K);

    return 0;
}
