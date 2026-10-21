#include "tile.h"
#include "expminus_fp16_spatz_params.h"

/* Unit-stride vector accesses must be 4-byte aligned. */
#define MISALIGNED(p)  (((uintptr_t)(p)) & 0x2)

/**
 * Computes a fast vectorized approximation of exp(x - m), m being a per-row scalar.
 * Uses the bit-level fast exponential method (Schraudolph 1999), as the exp kernel:
 *      exp(x) ~ reinterpret_fp16(COEF * x + BIAS)
 * COEF = 2^mantissa / ln(2) (mantissa = 10 for FP16), scales x to the FP16 exponent range.
 * BIAS = exponent_bias * 2^mantissa (bias = 15 for FP16), shifts the bits to approximate exp.
 * Assumes m >= x (m is the row max, as in softmax): x - m is clamped from below only.
 */
#define EXP_MIN  (-10.0f)
#define EXP_COEF (1477.0f)
#define EXP_BIAS (15360.0f)

/* v0 = reinterpret_fp16(COEF * max(v0, MIN) + BIAS), with v0 = x - m on entry. */
static inline void fast_exp_v0(void)
{
    register _Float16 min asm("f8")   = EXP_MIN;
    register _Float16 coef asm("f10") = EXP_COEF;
    register _Float16 bias asm("f11") = EXP_BIAS;

    asm volatile("vfmax.vf v0, v0, %0" ::"f"(min));
    asm volatile("vfmul.vf v0, v0, %0" ::"f"(coef));
    asm volatile("vfadd.vf v0, v0, %0" ::"f"(bias));
    asm volatile("vfcvt.rtz.xu.f.v v0, v0");
}

/* One row, vectorized along cols. A misaligned row (odd cols) peels its first element. */
static inline void expminus_row(const _Float16 *src, _Float16 *dst, size_t len, _Float16 m)
{
    _Float16 first;
    size_t vl;

    if (MISALIGNED(src)) {
        asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(1));
        asm volatile("vfmv.s.f v0, %0" ::"f"(*src));
        asm volatile("vfsub.vf v0, v0, %0" ::"f"(m));
        fast_exp_v0();
        asm volatile("vfmv.f.s %0, v0" : "=f"(first));
        *dst++ = first;
        src++;
        len--;
    }

    for (; len > 0; len -= vl, src += vl, dst += vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(len));
        asm volatile("vle16.v v0, (%0)" ::"r"(src));
        asm volatile("vfsub.vf v0, v0, %0" ::"f"(m));
        fast_exp_v0();
        asm volatile("vse16.v v0, (%0)" ::"r"(dst) : "memory");
    }
}

int expminus_fp16_spatz_task(void)
{
    volatile expminus_fp16_spatz_params_t *params;
    uintptr_t params_addr;
    const _Float16 *X;
    const _Float16 *M;
    _Float16 *Y;
    size_t cols;
    size_t rows;

    params_addr = mmio32(SPATZ_DATA);
    params      = (volatile expminus_fp16_spatz_params_t *)params_addr;

    X    = (const _Float16 *)params->shard_X;
    M    = (const _Float16 *)params->shard_M;
    Y    = (_Float16 *)params->shard_Y;
    cols = params->cols;
    rows = params->row_len;

    if (rows == 0 || cols == 0)
        return 0;

    /* Peeling aligns X and Y together only if they share alignment. */
    if (MISALIGNED((uintptr_t)X ^ (uintptr_t)Y))
        return 1;

    for (size_t r = 0; r < rows; r++)
        expminus_row(X + r * cols, Y + r * cols, cols, M[r]);

    return 0;
}
