#include "tile.h"
#include "expminus_rowsum_fp16_spatz_params.h"

/* Unit-stride vector accesses must be 4-byte aligned. */
#define MISALIGNED(p)  (((uintptr_t)(p)) & 0x2)

/* Fuses expminus (with the scale folded into the exp), reducesum over the last axis and
   exp of the max difference, with the same arithmetic as the separate kernels.
   exp(s * x) uses COEF * s and the clamps divided by s, precomputed by the caller: exact if s
   is a power of 2. */

/* v0 = reinterpret_fp16(coef * max(v0, min) + BIAS). */
static inline void fast_exp_v0(_Float16 min, _Float16 coef)
{
    register _Float16 bias asm("f11") = EXPMINUS_ROWSUM_EXP_BIAS;

    asm volatile(SPATZ_FPU_FENCE "vfmax.vf v0, v0, %0" ::"f"(min));
    asm volatile("vfmul.vf v0, v0, %0" ::"f"(coef));
    asm volatile("vfadd.vf v0, v0, %0" ::"f"(bias));
    asm volatile(SPATZ_FPU_FENCE "vfcvt.rtz.xu.f.v v0, v0");
}

/* One row, vectorized along cols. A misaligned row (odd cols) peels its first element.
   Returns *m_next, loaded before the row's store: a scalar load stalls on a pending vector
   store, which would drain the VU at every row. */
static inline _Float16 expminus_row(const _Float16 *src,
                                    _Float16 *dst,
                                    size_t len,
                                    _Float16 m,
                                    const _Float16 *m_next,
                                    _Float16 min,
                                    _Float16 coef)
{
    _Float16 first;
    _Float16 next;
    size_t vl;

    if (MISALIGNED(src)) {
        asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(1));
        asm volatile("vfmv.s.f v0, %0" ::"f"(*src));
        asm volatile("vfsub.vf v0, v0, %0" ::"f"(m));
        fast_exp_v0(min, coef);
        asm volatile("vfmv.f.s %0, v0" : "=f"(first));
        *dst++ = first;
        src++;
        len--;
    }

    if (len == 0)
        return *m_next;

    for (; len > 0; len -= vl, src += vl, dst += vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(len));
        asm volatile("vle16.v v0, (%0)" ::"r"(src));
        asm volatile("vfsub.vf v0, v0, %0" ::"f"(m));
        fast_exp_v0(min, coef);
        asm volatile("flh %0, 0(%1)" : "=f"(next) : "r"(m_next));
        asm volatile("vse16.v v0, (%0)" SPATZ_VST_FENCE ::"r"(dst) : "memory");
    }

    return next;
}

/* One row of expminus_rows_1strip, as two halves of h elements in v0 and v4 (e16, m4), its
   row sum reduced in FP32 into element 0 of `s`. Loads the next row max into m before the
   row's stores: a scalar load stalls on a pending vector store, which would drain the VU. */
#define EXPMINUS_ROW(s)                                                        \
    do {                                                                       \
        M = M < m_last ? M + 1 : M;                                            \
        asm volatile("vle16.v v0, (%0)" ::"r"(X));                             \
        asm volatile("vle16.v v4, (%0)" ::"r"(X + h));                         \
        asm volatile("vfsub.vf v0, v0, %0" ::"f"(m));                          \
        asm volatile("vfsub.vf v4, v4, %0" ::"f"(m));                          \
        asm volatile("vfmadd.vf v0, %0, v8" ::"f"(coef));                      \
        asm volatile("vfmadd.vf v4, %0, v8" ::"f"(coef));                      \
        asm volatile(SPATZ_FPU_FENCE "vfcvt.rtz.xu.f.v v0, v0");               \
        asm volatile("vfcvt.rtz.xu.f.v v4, v4");                               \
        asm volatile("flh %0, 0(%1)" : "=f"(m) : "r"(M));                      \
        asm volatile("vse16.v v0, (%0)" ::"r"(Y) : "memory");                  \
        asm volatile("vse16.v v4, (%0)" SPATZ_VST_FENCE ::"r"(Y + h) : "memory");              \
        asm volatile("vfwadd.vv v16, v0, v4");                                 \
        asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(h));            \
        asm volatile("vfredusum.vs " s ", v16, v31");                          \
        asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(h));            \
        X += cols;                                                             \
        Y += cols;                                                             \
    } while (0)

/* Stores the FP32 row sum in element 0 of `s` to `p` as FP16 (e16, vl = 1). */
#define EXPMINUS_SUM_ST(s, p)                                                  \
    do {                                                                       \
        asm volatile(SPATZ_FPU_FENCE "vfncvt.f.f.w v0, " s);                   \
        asm volatile("vse16.v v0, (%0)" SPATZ_VST_FENCE ::"r"(p) : "memory");                  \
    } while (0)

/* expminus_row and rowsum on all rows, for aligned rows of 2 h elements, h at most VLMAX at
   LMUL 4. The exp is coef * (x - m) + BIAS in one vfmadd against a BIAS splat, without the
   lower clamp: a negative result converts to 0, the limit of the exp. The row sum adds the two
   halves of the row while widened (in the same vtype as the exp), then reduces them in FP32
   with one vfredusum; the sums of 4 rows go to v24, v26, v28, v30 and are stored after one
   vtype switch (each switch drains the VU). No column-strided loads, which hit a single L1 bank
   when the row pitch is a multiple of 128 B. */
static inline void expminus_rows_1strip(const _Float16 *X,
                                        _Float16 *Y,
                                        _Float16 *S,
                                        const _Float16 *M,
                                        size_t rows,
                                        size_t cols,
                                        _Float16 coef)
{
    register _Float16 bias asm("f11") = EXPMINUS_ROWSUM_EXP_BIAS;
    const _Float16 *m_last = M + rows - 1;
    _Float16 m             = *M;
    size_t h               = cols / 2;
    size_t r               = 0;

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(1));
    asm volatile("vmv.s.x v31, zero");
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(h));
    asm volatile("vfmv.v.f v8, %0" ::"f"(bias));

    for (; r + 4 <= rows; r += 4) {
        EXPMINUS_ROW("v24");
        EXPMINUS_ROW("v26");
        EXPMINUS_ROW("v28");
        EXPMINUS_ROW("v30");
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(1));
        EXPMINUS_SUM_ST("v24", S + r + 0);
        EXPMINUS_SUM_ST("v26", S + r + 1);
        EXPMINUS_SUM_ST("v28", S + r + 2);
        EXPMINUS_SUM_ST("v30", S + r + 3);
        asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(h));
    }

    for (; r < rows; r++) {
        EXPMINUS_ROW("v24");
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(1));
        EXPMINUS_SUM_ST("v24", S + r);
        asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(h));
    }
}

/* S[r] = sum_c Y[r, c], one lane per row, fp32 accumulation (as reducesum). */
static inline void rowsum(const _Float16 *Y, _Float16 *S, size_t rows, size_t cols)
{
    register _Float16 zero asm("fs0") = 0.0f;
    const size_t row_stride = cols * sizeof(_Float16);
    size_t vl;

    for (size_t r = 0; r < rows; r += vl) {
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(rows - r));

        asm volatile("vlse16.v v8, (%0), %1" SPATZ_VLD_FENCE ::"r"(Y + r * cols), "r"(row_stride));
        asm volatile("vfwadd.vf v0, v8, %0" ::"f"(zero));
        for (size_t c = 1; c < cols; c++) {
            asm volatile("vlse16.v v8, (%0), %1" SPATZ_VLD_FENCE ::"r"(Y + r * cols + c), "r"(row_stride));
            asm volatile("vfwadd.wv v0, v0, v8");
        }

        asm volatile(SPATZ_FPU_FENCE "vfncvt.f.f.w v8, v0");
        asm volatile("vse16.v v8, (%0)" SPATZ_VST_FENCE ::"r"(S + r) : "memory");
    }
}

/* A[r] = exp(s * (A[r] - M[r])), with the clamps of the exp kernel. */
static inline void exp_diff(_Float16 *A, const _Float16 *M, size_t rows, _Float16 min,
                            _Float16 max, _Float16 coef)
{
    size_t vl;

    for (size_t r = 0; r < rows; r += vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(rows - r));
        asm volatile("vle16.v v0, (%0)" ::"r"(A + r));
        asm volatile("vle16.v v8, (%0)" ::"r"(M + r));
        asm volatile("vfsub.vv v0, v0, v8");
        asm volatile(SPATZ_FPU_FENCE "vfmin.vf v0, v0, %0" ::"f"(max));
        fast_exp_v0(min, coef);
        asm volatile("vse16.v v0, (%0)" SPATZ_VST_FENCE ::"r"(A + r) : "memory");
    }
}

/* The whole kernel, for any rows. Cold: linked out of the hot tasks, and the task does not
   spill callee-saved registers for it. */
static __attribute__((noinline, cold)) int expminus_rowsum_any(const _Float16 *X,
                                                              _Float16 *Y,
                                                              const _Float16 *M,
                                                              _Float16 *S,
                                                              _Float16 *A,
                                                              size_t rows,
                                                              size_t cols,
                                                              _Float16 min,
                                                              _Float16 max,
                                                              _Float16 coef)
{
    _Float16 m = M[0];

    for (size_t r = 0; r < rows; r++) {
        const _Float16 *m_next = M + (r + 1 < rows ? r + 1 : r);

        m = expminus_row(X + r * cols, Y + r * cols, cols, m, m_next, min, coef);
    }

    rowsum(Y, S, rows, cols);

    if (A)
        exp_diff(A, M, rows, min, max, coef);

    return 0;
}

int expminus_rowsum_fp16_spatz_task(void)
{
    const expminus_rowsum_fp16_spatz_params_t *params;
    const _Float16 *X;
    const _Float16 *M;
    _Float16 *Y;
    _Float16 *S;
    _Float16 *A;
    union { uint32_t u; _Float16 f[2]; } min_max, coef;
    _Float16 min, max;
    size_t rows;
    size_t cols;

    params = (const expminus_rowsum_fp16_spatz_params_t *)mmio32(SPATZ_DATA);

    X          = (const _Float16 *)params->shard_X;
    M          = (const _Float16 *)params->shard_M;
    Y          = (_Float16 *)params->shard_Y;
    S          = (_Float16 *)params->shard_S;
    A          = (_Float16 *)params->shard_A;
    rows       = params->dims & 0xffff;
    cols       = params->dims >> 16;
    min_max.u  = params->min_max;
    coef.u     = params->coef;
    min        = min_max.f[0];
    max        = min_max.f[1];

    if (rows == 0 || cols == 0)
        return 0;

    /* Peeling aligns X and Y together only if they share alignment. */
    if (MISALIGNED((uintptr_t)X ^ (uintptr_t)Y) || MISALIGNED(S) || MISALIGNED(A) ||
        MISALIGNED(M))
        return 1;

    size_t vl;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(cols / 2));
    if (vl != cols / 2 || MISALIGNED(X) || (cols & 3))
        return expminus_rowsum_any(X, Y, M, S, A, rows, cols, min, max, coef.f[0]);

    expminus_rows_1strip(X, Y, S, M, rows, cols, coef.f[0]);

    if (A)
        exp_diff(A, M, rows, min, max, coef.f[0]);

    return 0;
}
