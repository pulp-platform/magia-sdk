#include "tile.h"
#include "rmsnorm_fp16_spatz_params.h"

/* Sum of squares of one row: the squares are exact in fp32 (vfwmacc, fp16 x fp16) and accumulate
   per lane in fp32, then one fp32 reduction. */
static inline float sum_squares(const _Float16 *src, size_t len)
{
    float ZERO = 0.0f;
    float sum;
    size_t avl;
    size_t vl;

    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(len));
    asm volatile("vfmv.v.f v8, %0" ::"f"(ZERO));
    asm volatile("vfmv.s.f v16, %0" ::"f"(ZERO));

    for (avl = len; avl > 0; avl -= vl) {
        /* tail undisturbed: the lanes past a short last chunk keep their partial sums */
        asm volatile("vsetvli %0, %1, e16, m4, tu, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle16.v v0, (%0)" ::"r"(src));
        asm volatile("vfwmacc.vv v8, v0, v0");
        src += vl;
    }

    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(len));
    asm volatile("vfredusum.vs v16, v8, v16");
    asm volatile("vfmv.f.s %0, v16" : "=f"(sum));

    return sum;
}

/* dst = src * gamma * rinv: src * gamma is exact in fp32 (vfwmul), one rounding to fp16. */
static inline void scale_row(const _Float16 *src,
                             const _Float16 *gamma,
                             _Float16 *dst,
                             float rinv,
                             size_t len)
{
    size_t avl;
    size_t vl;

    for (avl = len; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile("vle16.v v0, (%0)" ::"r"(src));
        asm volatile("vle16.v v4, (%0)" ::"r"(gamma));
        asm volatile("vfwmul.vv v8, v0, v4");
        asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(vl));
        asm volatile("vfmul.vf v8, v8, %0" ::"f"(rinv));
        asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(vl));
        asm volatile("vfncvt.f.f.w v0, v8");
        asm volatile("vse16.v v0, (%0)" ::"r"(dst) : "memory");
        src += vl;
        gamma += vl;
        dst += vl;
    }
}

int rmsnorm_fp16_spatz_task(void)
{
    volatile rmsnorm_fp16_spatz_params_t *params;
    const _Float16 *gamma;
    const _Float16 *X;
    _Float16 *Y;
    float eps;
    size_t r_len;
    size_t w_len;

    params = (volatile rmsnorm_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    X      = (const _Float16 *)params->shard_X;
    Y      = (_Float16 *)params->shard_Y;
    gamma  = (const _Float16 *)params->gamma;
    eps    = (float)*(_Float16 *)params->eps;
    r_len  = params->r_len;
    w_len  = params->w_len;

    for (size_t r = 0; r < r_len; r++) {
        float ms;
        float rinv;

        ms = sum_squares(X + r * w_len, w_len) / (float)w_len + eps;
        asm volatile("fsqrt.s %0, %1" : "=f"(rinv) : "f"(ms));
        rinv = 1.0f / rinv;

        scale_row(X + r * w_len, gamma, Y + r * w_len, rinv, w_len);
    }

    return 0;
}
