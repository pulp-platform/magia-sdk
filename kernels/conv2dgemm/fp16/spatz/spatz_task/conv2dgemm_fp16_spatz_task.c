#include "tile.h"
#include "conv2dgemm_fp16_spatz_params.h"

/* Uncomment to accumulate the GEMM in fp32 (widening MAC + narrow to fp16) instead of fp16. */
// #define FP32_ACCUM

static inline void
im2col_copy(const _Float16 *src, _Float16 *dst, const size_t len, const size_t stride)
{
    const _Float16 *p_src;
    _Float16 *p_dst;
    size_t stride_bytes;
    size_t avl;
    size_t vl;

    p_src        = src;
    p_dst        = dst;
    avl          = len;
    stride_bytes = stride * sizeof(_Float16);

    if (stride == 1) {
        for (; avl > 0; avl -= vl) {
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
            asm volatile("vle16.v v0, (%0)" ::"r"(p_src));
            asm volatile("vse16.v v0, (%0)" ::"r"(p_dst) : "memory");
            p_src += vl;
            p_dst += vl;
        }
    } else {
        for (; avl > 0; avl -= vl) {
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
            asm volatile("vlse16.v v0, (%0), %1" ::"r"(p_src), "r"(stride_bytes));
            asm volatile("vse16.v v0, (%0)" ::"r"(p_dst) : "memory");
            p_src += vl * stride;
            p_dst += vl;
        }
    }
}

static inline void im2col_zero(_Float16 *dst, const size_t len)
{
    register _Float16 ZERO asm("fs0") = 0.0f;
    _Float16 *p_dst;
    size_t avl;
    size_t vl;

    p_dst = dst;
    avl   = len;

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vfmv.v.f v0, %0" ::"f"(ZERO));
        asm volatile("vse16.v v0, (%0)" ::"r"(p_dst) : "memory");
        p_dst += vl;
    }
}

static void im2col(const _Float16 *X,
                   _Float16 *B,
                   const uint32_t n_batches,
                   const uint32_t c_in,
                   const uint32_t h_in,
                   const uint32_t w_in,
                   const uint32_t w_out,
                   const uint32_t kernel_h,
                   const uint32_t kernel_w,
                   const uint32_t stride_h,
                   const uint32_t stride_w,
                   const uint32_t pad_h,
                   const uint32_t pad_w,
                   const uint32_t oh_start,
                   const uint32_t oh_len,
                   const uint32_t ic_start,
                   const uint32_t ic_end)
{
    uint32_t hw_out = oh_len * w_out;
    uint32_t in_chw = c_in * h_in * w_in;

    /* im2col over this tile's output rows: B[b][k, n] = X[b, ic, ih, iw] (0 if padded) */
    for (uint32_t b = 0; b < n_batches; b++) {
        const _Float16 *X_b = X + b * in_chw;
        _Float16 *B_b       = B + b * (c_in * kernel_h * kernel_w) * hw_out;

        for (uint32_t ic = ic_start; ic < ic_end; ic++) {
            const _Float16 *x_c = X_b + ic * h_in * w_in;
            for (uint32_t ki = 0; ki < kernel_h; ki++) {
                for (uint32_t kj = 0; kj < kernel_w; kj++) {
                    uint32_t k = (ic * kernel_h + ki) * kernel_w + kj;

                    /* Output columns inside the image: [ow_start, ow_end) */
                    int iw_off   = (int)kj - (int)pad_w;
                    int last     = (int)w_in - 1 - iw_off;
                    int ow_start = iw_off < 0 ? (-iw_off + (int)stride_w - 1) / (int)stride_w : 0;
                    int ow_end   = last < 0 ? 0 : last / (int)stride_w + 1;
                    if (ow_end > (int)w_out)
                        ow_end = (int)w_out;
                    if (ow_start > ow_end)
                        ow_start = ow_end;

                    for (uint32_t oh = 0; oh < oh_len; oh++) {
                        int ih          = (int)((oh_start + oh) * stride_h) - (int)pad_h + (int)ki;
                        _Float16 *B_row = B_b + k * hw_out + oh * w_out;

                        if (ih < 0 || ih >= (int)h_in) {
                            im2col_zero(B_row, w_out);
                            continue;
                        }

                        const _Float16 *x_row = x_c + (uint32_t)ih * w_in;
                        if (ow_start > 0)
                            im2col_zero(B_row, ow_start);
                        if (ow_end > ow_start)
                            im2col_copy(x_row + ow_start * (int)stride_w + iw_off,
                                        B_row + ow_start,
                                        ow_end - ow_start,
                                        stride_w);
                        if (ow_end < (int)w_out)
                            im2col_zero(B_row + ow_end, w_out - ow_end);
                    }
                }
            }
        }
    }
}

static void gemm(const _Float16 *A,
                 const _Float16 *B,
                 const _Float16 *C,
                 _Float16 *Y,
                 _Float16 alpha,
                 _Float16 beta,
                 const size_t dim_M,
                 const size_t dim_N,
                 const size_t dim_K)
{
    register _Float16 ZERO asm("fs0") = 0.0f;
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
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        /* 2) Unroll over two A-col at time (two B-rows at time) */
        for (int m = 0; m < (dim_M - 1); m += 2) {
            C_row1 = C + (m * dim_N + n);
            Y_row1 = Y + (m * dim_N + n);
            C_row2 = C + ((m + 1) * dim_N + n);
            Y_row2 = Y + ((m + 1) * dim_N + n);

            asm volatile("vfmv.v.f v0, %0" ::"f"(ZERO));
            asm volatile("vfmv.v.f v8, %0" ::"f"(ZERO));

            if (alpha != 0.0f) {
                /* 1) Unroll over two elements of each A-col at time (same B-row) */
                for (int k = 0; k < (int)dim_K - 1; k += 2) {
                    B_row1       = B + (k * dim_N + n);
                    B_row2       = B + ((k + 1) * dim_N + n);
                    A_col1_elem1 = A + (m * dim_K + k);
                    A_col2_elem1 = A + (m * dim_K + (k + 1));
                    A_col1_elem2 = A + ((m + 1) * dim_K + k);
                    A_col2_elem2 = A + ((m + 1) * dim_K + (k + 1));

                    asm volatile("vle16.v v16, (%0)" ::"r"(B_row1));
                    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(*A_col1_elem1));
                    asm volatile("vfmacc.vf v8, %0, v16" ::"f"(*A_col1_elem2));

                    asm volatile("vle16.v v24, (%0)" ::"r"(B_row2));
                    asm volatile("vfmacc.vf v0, %0, v24" ::"f"(*A_col2_elem1));
                    asm volatile("vfmacc.vf v8, %0, v24" ::"f"(*A_col2_elem2));
                }

                /* Leftovers for elements of each A-col (same B-rows) */
                if (dim_K % 2) {
                    B_row1       = B + ((dim_K - 1) * dim_N + n);
                    A_col1_elem1 = A + (m * dim_K + (dim_K - 1));
                    A_col1_elem2 = A + ((m + 1) * dim_K + (dim_K - 1));

                    asm volatile("vle16.v v16, (%0)" ::"r"(B_row1));
                    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(*A_col1_elem1));
                    asm volatile("vfmacc.vf v8, %0, v16" ::"f"(*A_col1_elem2));
                }

                /* acc = alpha * A @ B */
                asm volatile("vfmul.vf v0, v0, %0" ::"f"(alpha));
                asm volatile("vfmul.vf v8, v8, %0" ::"f"(alpha));
            }

            /* acc += beta * C */
            if (beta != 0.0f) {
                asm volatile("vle16.v v16, (%0)" ::"r"(C_row1));
                asm volatile("vle16.v v24, (%0)" ::"r"(C_row2));
                asm volatile("vfmacc.vf v0, %0, v16" ::"f"(beta));
                asm volatile("vfmacc.vf v8, %0, v24" ::"f"(beta));
            }

            asm volatile("vse16.v v0, (%0)" ::"r"(Y_row1));
            asm volatile("vse16.v v8, (%0)" SPATZ_VST_FENCE ::"r"(Y_row2));
        }

        /* Leftovers for A-cols and B-rows  */
        if (dim_M % 2) {
            C_row1 = C + ((dim_M - 1) * dim_N + n);
            Y_row1 = Y + ((dim_M - 1) * dim_N + n);

            asm volatile("vfmv.v.f v0, %0" ::"f"(ZERO));

            if (alpha != 0.0f) {
                for (int k = 0; k < (int)dim_K - 1; k += 2) {
                    B_row1       = B + (k * dim_N + n);
                    B_row2       = B + ((k + 1) * dim_N + n);
                    A_col1_elem1 = A + ((dim_M - 1) * dim_K + k);
                    A_col2_elem1 = A + ((dim_M - 1) * dim_K + (k + 1));

                    asm volatile("vle16.v v16, (%0)" ::"r"(B_row1));
                    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(*A_col1_elem1));

                    asm volatile("vle16.v v24, (%0)" ::"r"(B_row2));
                    asm volatile("vfmacc.vf v0, %0, v24" ::"f"(*A_col2_elem1));
                }

                if (dim_K % 2) {
                    B_row1       = B + ((dim_K - 1) * dim_N + n);
                    A_col1_elem1 = A + ((dim_M - 1) * dim_K + (dim_K - 1));

                    asm volatile("vle16.v v16, (%0)" ::"r"(B_row1));
                    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(*A_col1_elem1));
                }

                /* acc = alpha * A @ B */
                asm volatile("vfmul.vf v0, v0, %0" ::"f"(alpha));
            }

            /* acc += beta * C */
            if (beta != 0.0f) {
                asm volatile("vle16.v v16, (%0)" ::"r"(C_row1));
                asm volatile("vfmacc.vf v0, %0, v16" ::"f"(beta));
            }

            asm volatile("vse16.v v0, (%0)" SPATZ_VST_FENCE ::"r"(Y_row1));
        }
    }
}

static void gemm_cols(const _Float16 *A,
                      const _Float16 *B,
                      const _Float16 *C,
                      _Float16 *Y,
                      _Float16 alpha,
                      _Float16 beta,
                      const size_t dim_M,
                      const size_t dim_N,
                      const size_t dim_K)
{
    register _Float16 ZERO asm("fs0") = 0.0f;
    const _Float16 *A_col;
    const _Float16 *C_col;
    _Float16 *Y_col;
    size_t stride_A;
    size_t stride_N;
    size_t avl;
    size_t vl;

    /* Few output pixels: vectorize along the output channels (same sum order as gemm()) */
    stride_A = dim_K * sizeof(_Float16);
    stride_N = dim_N * sizeof(_Float16);

    for (int m = 0; m < dim_M; m += vl) {
        avl = dim_M - m;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        for (int n = 0; n < dim_N; n++) {
            C_col = C + (m * dim_N + n);
            Y_col = Y + (m * dim_N + n);

            asm volatile("vfmv.v.f v0, %0" ::"f"(ZERO));

            if (alpha != 0.0f) {
                for (int k = 0; k < (int)dim_K; k++) {
                    A_col = A + (m * dim_K + k);
                    asm volatile("vlse16.v v16, (%0), %1" ::"r"(A_col), "r"(stride_A));
                    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(B[k * dim_N + n]));
                }

                /* acc = alpha * A @ B */
                asm volatile("vfmul.vf v0, v0, %0" ::"f"(alpha));
            }

            /* acc += beta * C */
            if (beta != 0.0f) {
                asm volatile("vlse16.v v16, (%0), %1" ::"r"(C_col), "r"(stride_N));
                asm volatile("vfmacc.vf v0, %0, v16" ::"f"(beta));
            }

            asm volatile("vsse16.v v0, (%0), %1" ::"r"(Y_col), "r"(stride_N));
        }
    }
}

static void gemm_grouped(const _Float16 *A,
                         const _Float16 *B,
                         const _Float16 *C,
                         _Float16 *Y,
                         _Float16 alpha,
                         _Float16 beta,
                         const size_t dim_M,
                         const size_t dim_N,
                         const size_t dim_Kg,
                         const size_t cout_g,
                         const size_t oc_start)
{
    register _Float16 ZERO asm("fs0") = 0.0f;
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
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        for (int m = 0; m < dim_M; m++) {
            int g = (oc_start + m) / cout_g;
            A_row = A + m * dim_Kg;
            B_grp = B + g * dim_Kg * dim_N;
            C_row = C + (m * dim_N + n);
            Y_row = Y + (m * dim_N + n);

            asm volatile("vfmv.v.f v0, %0" ::"f"(ZERO));

            if (alpha != 0.0f) {
                for (int k = 0; k < (int)dim_Kg; k++) {
                    B_row = B_grp + (k * dim_N + n);
                    asm volatile("vle16.v v16, (%0)" ::"r"(B_row));
                    asm volatile("vfmacc.vf v0, %0, v16" ::"f"(A_row[k]));
                }

                asm volatile("vfmul.vf v0, v0, %0" ::"f"(alpha));
            }

            if (beta != 0.0f) {
                asm volatile("vle16.v v16, (%0)" ::"r"(C_row));
                asm volatile("vfmacc.vf v0, %0, v16" ::"f"(beta));
            }

            asm volatile("vse16.v v0, (%0)" SPATZ_VST_FENCE ::"r"(Y_row));
        }
    }
}

#ifdef FP32_ACCUM

static void gemm_accfp32(const _Float16 *A,
                         const _Float16 *B,
                         const _Float16 *C,
                         _Float16 *Y,
                         _Float16 alpha,
                         _Float16 beta,
                         const size_t dim_M,
                         const size_t dim_N,
                         const size_t dim_K)
{
    register _Float16 ZERO asm("fs0") = 0.0f;
    const _Float16 *B_row;
    const _Float16 *C_row;
    _Float16 *Y_row;
    size_t avl;
    size_t vl;

    /* Each output row is a dot product over dim_K accumulated in fp32: v8 (e32) is built with a
       widening multiply on the first term and widening MACs on the rest, then narrowed to fp16. */
    for (int n = 0; n < dim_N; n += vl) {
        avl = dim_N - n;
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl));

        for (int m = 0; m < dim_M; m++) {
            C_row = C + (m * dim_N + n);
            Y_row = Y + (m * dim_N + n);

            if (alpha != 0.0f) {
                B_row = B + n;
                asm volatile("vle16.v v0, (%0)" ::"r"(B_row));
                asm volatile("vfwmul.vf v8, v0, %0" ::"f"(*(A + m * dim_K)));

                for (int k = 1; k < (int)dim_K; k++) {
                    B_row = B + (k * dim_N + n);
                    asm volatile("vle16.v v0, (%0)" ::"r"(B_row));
                    asm volatile("vfwmacc.vf v8, %0, v0" ::"f"(*(A + m * dim_K + k)));
                }

                asm volatile("vfncvt.f.f.w v16, v8");
                asm volatile("vfmul.vf v16, v16, %0" ::"f"(alpha));
            } else {
                asm volatile("vfmv.v.f v16, %0" ::"f"(ZERO));
            }

            if (beta != 0.0f) {
                asm volatile("vle16.v v0, (%0)" ::"r"(C_row));
                asm volatile("vfmacc.vf v16, %0, v0" ::"f"(beta));
            }

            asm volatile("vse16.v v16, (%0)" SPATZ_VST_FENCE ::"r"(Y_row));
        }
    }
}

static void gemm_grouped_accfp32(const _Float16 *A,
                                 const _Float16 *B,
                                 const _Float16 *C,
                                 _Float16 *Y,
                                 _Float16 alpha,
                                 _Float16 beta,
                                 const size_t dim_M,
                                 const size_t dim_N,
                                 const size_t dim_Kg,
                                 const size_t cout_g,
                                 const size_t oc_start)
{
    register _Float16 ZERO asm("fs0") = 0.0f;
    const _Float16 *A_row;
    const _Float16 *B_grp;
    const _Float16 *B_row;
    const _Float16 *C_row;
    _Float16 *Y_row;
    size_t avl;
    size_t vl;

    for (int n = 0; n < dim_N; n += vl) {
        avl = dim_N - n;
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl));

        for (int m = 0; m < dim_M; m++) {
            int g = (oc_start + m) / cout_g;
            A_row = A + m * dim_Kg;
            B_grp = B + g * dim_Kg * dim_N;
            C_row = C + (m * dim_N + n);
            Y_row = Y + (m * dim_N + n);

            if (alpha != 0.0f) {
                B_row = B_grp + n;
                asm volatile("vle16.v v0, (%0)" ::"r"(B_row));
                asm volatile("vfwmul.vf v8, v0, %0" ::"f"(A_row[0]));

                for (int k = 1; k < (int)dim_Kg; k++) {
                    B_row = B_grp + (k * dim_N + n);
                    asm volatile("vle16.v v0, (%0)" ::"r"(B_row));
                    asm volatile("vfwmacc.vf v8, %0, v0" ::"f"(A_row[k]));
                }

                asm volatile("vfncvt.f.f.w v16, v8");
                asm volatile("vfmul.vf v16, v16, %0" ::"f"(alpha));
            } else {
                asm volatile("vfmv.v.f v16, %0" ::"f"(ZERO));
            }

            if (beta != 0.0f) {
                asm volatile("vle16.v v0, (%0)" ::"r"(C_row));
                asm volatile("vfmacc.vf v16, %0, v0" ::"f"(beta));
            }

            asm volatile("vse16.v v16, (%0)" SPATZ_VST_FENCE ::"r"(Y_row));
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
    _Float16 *X;
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
    uint32_t c_in, h_in, w_in, w_out, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w;
    uint32_t oh_start, oh_len;
    uint32_t c_in_g, ic_start, ic_end;

    params_addr = mmio32(SPATZ_DATA);
    params      = (volatile conv2dgemm_fp16_spatz_params_t *)params_addr;

    alpha     = *(_Float16 *)params->alpha;
    beta      = *(_Float16 *)params->beta;
    X         = (_Float16 *)params->shard_X;
    A         = (_Float16 *)params->shard_A;
    B         = (_Float16 *)params->shard_B;
    C         = (_Float16 *)params->shard_C;
    Y         = (_Float16 *)params->shard_Y;
    n_batches = params->n_batches;
    oc_start  = params->oc_start;
    group     = params->group;
    c_out     = params->c_out;
    M         = params->M;
    N         = params->N;
    K         = params->K;
    c_in      = params->c_in;
    h_in      = params->h_in;
    w_in      = params->w_in;
    w_out     = params->w_out;
    kernel_h  = params->kernel_h;
    kernel_w  = params->kernel_w;
    stride_h  = params->stride_h;
    stride_w  = params->stride_w;
    pad_h     = params->pad_h;
    pad_w     = params->pad_w;
    oh_start  = params->oh_start;
    oh_len    = params->oh_len;

    if (M == 0)
        return 0;

    K_g    = K / group;
    cout_g = c_out / group;

    /* Only the rows of B of this tile's groups; with B == X (1x1) nothing to build */
    if (B != X) {
        c_in_g   = c_in / group;
        ic_start = (oc_start / cout_g) * c_in_g;
        ic_end   = ((oc_start + M - 1) / cout_g + 1) * c_in_g;

        im2col(X,
               B,
               n_batches,
               c_in,
               h_in,
               w_in,
               w_out,
               kernel_h,
               kernel_w,
               stride_h,
               stride_w,
               pad_h,
               pad_w,
               oh_start,
               oh_len,
               ic_start,
               ic_end);
    }

    /* Weights (A) and bias (C) are shared across batches; the im2col matrix (B) and the output
       (Y) are batched, so advance them by one 2D slice per batch. group == 1 is a plain dense
       GEMM; grouped/depthwise convs apply each output channel's compact weights to only its
       group's block of B.  */
    for (size_t b = 0; b < n_batches; b++) {
#ifdef FP32_ACCUM
        if (group == 1)
            gemm_accfp32(A, B + b * (K * N), C, Y + b * (M * N), alpha, beta, M, N, K_g);
        else
            gemm_grouped_accfp32(
                A, B + b * (K * N), C, Y + b * (M * N), alpha, beta, M, N, K_g, cout_g, oc_start);
#else
        /* Fewer pixels than channels: gemm_cols() issues fewer vector instructions */
        if (group == 1 && 4 * N < 3 * M && N <= 128)
            gemm_cols(A, B + b * (K * N), C, Y + b * (M * N), alpha, beta, M, N, K_g);
        else if (group == 1)
            gemm(A, B + b * (K * N), C, Y + b * (M * N), alpha, beta, M, N, K_g);
        else
            gemm_grouped(
                A, B + b * (K * N), C, Y + b * (M * N), alpha, beta, M, N, K_g, cout_g, oc_start);
#endif
    }

    return 0;
}
