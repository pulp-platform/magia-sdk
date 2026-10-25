#include "tile.h"
#include "bias_rope_fp16_spatz_params.h"

/* Unit-stride vector accesses must be 4-byte aligned. */
#define MISALIGNED(p)  (((uintptr_t)(p)) & 0x2)

/* X += bias (fp16 adds), broadcast along the rows; then, if cos is given, the RoPE of each row
   (token) with the channels of a head interleaved, pair i = columns (2i, 2i + 1) rotated by angle i:
     y[2i]     = x[2i] * cos[i] - x[2i + 1] * sin[i]
     y[2i + 1] = x[2i + 1] * cos[i] + x[2i] * sin[i]
   as the rotate-half RoPE on the columns permuted per head (column i <- i, P/2 + i <- i): Q and K
   permuted the same way give the same Q @ Kt. Even and odd columns by strided accesses; the four
   products are exact in fp32 (vfwmul, fp16 x fp16), the sum and difference are rounded once in
   fp32 and once to fp16 (as kernels/rope). The block may start inside a head (col0) and straddle
   heads: each row is processed one head segment at a time. */
int bias_rope_fp16_spatz_task(void)
{
    const bias_rope_fp16_spatz_params_t *params;
    _Float16 *X;
    const _Float16 *B;
    const _Float16 *C;
    const _Float16 *S;
    size_t rows;
    size_t cols;
    size_t half;
    size_t f0;
    size_t vl;

    params = (const bias_rope_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    X      = (_Float16 *)params->shard_X;
    B      = (const _Float16 *)params->bias;
    C      = (const _Float16 *)params->cos;
    S      = (const _Float16 *)params->sin;
    rows   = params->dims & 0xffff;
    cols   = params->dims >> 16;
    half   = params->head_dim / 2;
    f0     = params->col0 / 2;

    if (!C) {
        for (size_t r = 0; r < rows; r++) {
            for (size_t c = 0; c < cols; c += vl) {
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(cols - c));
                asm volatile("vle16.v v0, (%0)" ::"r"(X + r * cols + c));
                asm volatile("vle16.v v8, (%0)" ::"r"(B + c));
                asm volatile("vfadd.vv v0, v0, v8");
                asm volatile("vse16.v v0, (%0)" SPATZ_VST_FENCE ::"r"(X + r * cols + c) : "memory");
            }
        }
        return 0;
    }

    for (size_t r = 0; r < rows; r++) {
        const _Float16 *c_row = C + r * half;
        const _Float16 *s_row = S + r * half;
        size_t f = f0;

        for (size_t p = 0; p < cols / 2;) {
            size_t seg = half - f < cols / 2 - p ? half - f : cols / 2 - p;

            for (size_t j = 0; j < seg; j += vl) {
                _Float16 *xe       = X + r * cols + 2 * (p + j);
                const _Float16 *be = B + 2 * (p + j);
                const _Float16 *cf = c_row + f + j;
                const _Float16 *sf = s_row + f + j;

                asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(vl) : "r"(seg - j));
                asm volatile("vlse16.v v0, (%0), %1" SPATZ_VLD_FENCE ::"r"(xe), "r"(4));
                asm volatile("vlse16.v v2, (%0), %1" SPATZ_VLD_FENCE ::"r"(xe + 1), "r"(4));
                asm volatile("vlse16.v v4, (%0), %1" SPATZ_VLD_FENCE ::"r"(be), "r"(4));
                asm volatile("vlse16.v v6, (%0), %1" SPATZ_VLD_FENCE ::"r"(be + 1), "r"(4));
                asm volatile("vfadd.vv v0, v0, v4"); /* x even + bias */
                asm volatile("vfadd.vv v2, v2, v6"); /* x odd + bias */
                if (MISALIGNED(cf)) {
                    asm volatile("vlse16.v v4, (%0), %1" SPATZ_VLD_FENCE ::"r"(cf), "r"(2));
                    asm volatile("vlse16.v v6, (%0), %1" SPATZ_VLD_FENCE ::"r"(sf), "r"(2));
                } else {
                    asm volatile("vle16.v v4, (%0)" ::"r"(cf));
                    asm volatile("vle16.v v6, (%0)" ::"r"(sf));
                }
                asm volatile("vfwmul.vv v8, v0, v4");  /* xe * cos */
                asm volatile("vfwmul.vv v12, v2, v6"); /* xo * sin */
                asm volatile("vfwmul.vv v16, v2, v4"); /* xo * cos */
                asm volatile("vfwmul.vv v20, v0, v6"); /* xe * sin */

                asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(vl));
                asm volatile("vfsub.vv v8, v8, v12");
                asm volatile("vfadd.vv v16, v16, v20");

                asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(vl));
                asm volatile(SPATZ_FPU_FENCE "vfncvt.f.f.w v24, v8");
                asm volatile("vfncvt.f.f.w v26, v16");
                asm volatile("vsse16.v v24, (%0), %1" ::"r"(xe), "r"(4) : "memory");
                asm volatile("vsse16.v v26, (%0), %1" SPATZ_VST_FENCE ::"r"(xe + 1), "r"(4) : "memory");
            }
            p += seg;
            f = 0;
        }
    }

    return 0;
}
