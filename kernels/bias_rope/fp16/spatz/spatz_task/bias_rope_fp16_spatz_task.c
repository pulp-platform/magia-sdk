#include "tile.h"
#include "bias_rope_fp16_spatz_params.h"

/* X += bias (fp16 adds), broadcast along the rows; then, if cos is given, the RoPE of each row
   (token) with the channels of a head in groups of G = group: in group g, columns
   a = g * G + j and b = a + G/2 (j < G/2) are the pair of angle i = g * G/2 + j,
     y[a] = x[a] * cos[i] - x[b] * sin[i]
     y[b] = x[b] * cos[i] + x[a] * sin[i]
   i.e. the rotate-half RoPE (pairs i, P/2 + i) on the channels permuted per head: Q and K permuted
   the same way give the same Q @ Kt. All the accesses are unit-stride, in vectors of up to G/2
   elements; the four products are exact in fp32 (vfwmul, fp16 x fp16), the sum and difference are
   rounded once in fp32 and once to fp16 (as kernels/rope). The block starts at column col0 of its
   head (a multiple of G) and may straddle heads; its columns are a multiple of G. */
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
    size_t gh;
    size_t i0;
    size_t vl;

    params = (const bias_rope_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    X      = (_Float16 *)params->shard_X;
    B      = (const _Float16 *)params->bias;
    C      = (const _Float16 *)params->cos;
    S      = (const _Float16 *)params->sin;
    rows   = params->dims & 0xffff;
    cols   = params->dims >> 16;
    half   = params->head_dim / 2;
    gh     = params->group / 2;
    i0     = params->col0 / 2;

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
        size_t i = i0; /* angle of the first pair of the group */

        for (size_t g0 = 0; g0 < cols; g0 += 2 * gh) {
            for (size_t j = 0; j < gh; j += vl) {
                _Float16 *xa       = X + r * cols + g0 + j;
                _Float16 *xb       = xa + gh;
                const _Float16 *ba = B + g0 + j;
                const _Float16 *bb = ba + gh;

                asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(vl) : "r"(gh - j));
                asm volatile("vle16.v v0, (%0)" ::"r"(xa));
                asm volatile("vle16.v v2, (%0)" ::"r"(xb));
                asm volatile("vle16.v v4, (%0)" ::"r"(ba));
                asm volatile("vle16.v v6, (%0)" ::"r"(bb));
                asm volatile("vfadd.vv v0, v0, v4"); /* x[a] + bias */
                asm volatile("vfadd.vv v2, v2, v6"); /* x[b] + bias */
                asm volatile("vle16.v v4, (%0)" ::"r"(c_row + i + j));
                asm volatile("vle16.v v6, (%0)" ::"r"(s_row + i + j));
                asm volatile("vfwmul.vv v8, v0, v4");  /* xa * cos */
                asm volatile("vfwmul.vv v12, v2, v6"); /* xb * sin */
                asm volatile("vfwmul.vv v16, v2, v4"); /* xb * cos */
                asm volatile("vfwmul.vv v20, v0, v6"); /* xa * sin */

                asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(vl));
                asm volatile("vfsub.vv v8, v8, v12");
                asm volatile("vfadd.vv v16, v16, v20");

                asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(vl));
                asm volatile(SPATZ_FPU_FENCE "vfncvt.f.f.w v24, v8");
                asm volatile("vfncvt.f.f.w v26, v16");
                asm volatile("vse16.v v24, (%0)" ::"r"(xa) : "memory");
                asm volatile("vse16.v v26, (%0)" SPATZ_VST_FENCE ::"r"(xb) : "memory");
            }
            i = i + gh == half ? 0 : i + gh;
        }
    }

    return 0;
}
