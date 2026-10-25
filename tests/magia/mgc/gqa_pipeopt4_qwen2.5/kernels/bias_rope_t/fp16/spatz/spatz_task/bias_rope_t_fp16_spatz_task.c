#include "tile.h"
#include "bias_rope_t_fp16_spatz_params.h"

/* bias_rope on a transposed block (channels along the rows, tokens along the columns, as Kt):
   X[r, t] += bias[r] (fp16 adds), then, in each group g of G = group rows of a head, rows
   a = g * G + j and b = a + G/2 (j < G/2) rotated by angle i = g * G/2 + j,
     y[a, t] = x[a, t] * cos[i, t] - x[b, t] * sin[i, t]
     y[b, t] = x[b, t] * cos[i, t] + x[a, t] * sin[i, t]
   with the same arithmetic as bias_rope (exact fp32 products, fp32 then fp16 rounding), all
   unit-stride along the tokens. The block starts at row row0 of its head (a multiple of G) and may
   straddle heads; its rows are a multiple of G. */
int bias_rope_t_fp16_spatz_task(void)
{
    const bias_rope_t_fp16_spatz_params_t *params;
    _Float16 *X;
    const _Float16 *B;
    const _Float16 *C;
    const _Float16 *S;
    size_t rows;
    size_t cols;
    size_t half;
    size_t gh;
    size_t i;
    size_t vl;

    params = (const bias_rope_t_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    X      = (_Float16 *)params->shard_X;
    B      = (const _Float16 *)params->bias;
    C      = (const _Float16 *)params->cos;
    S      = (const _Float16 *)params->sin;
    rows   = params->dims & 0xffff;
    cols   = params->dims >> 16;
    half   = params->head_dim / 2;
    gh     = params->group / 2;
    i      = params->row0 / 2; /* angle of the first pair of the group */

    for (size_t g0 = 0; g0 < rows; g0 += 2 * gh) {
        for (size_t j = 0; j < gh; j++) {
            _Float16 *xa       = X + (g0 + j) * cols;
            _Float16 *xb       = xa + gh * cols;
            const _Float16 *cf = C + (i + j) * cols;
            const _Float16 *sf = S + (i + j) * cols;
            _Float16 ba        = B[g0 + j];
            _Float16 bb        = B[g0 + j + gh];

            for (size_t t = 0; t < cols; t += vl) {
                asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(vl) : "r"(cols - t));
                asm volatile("vle16.v v0, (%0)" ::"r"(xa + t));
                asm volatile("vle16.v v2, (%0)" ::"r"(xb + t));
                asm volatile("vle16.v v4, (%0)" ::"r"(cf + t));
                asm volatile("vle16.v v6, (%0)" ::"r"(sf + t));
                asm volatile("vfadd.vf v0, v0, %0" ::"f"(ba));
                asm volatile("vfadd.vf v2, v2, %0" ::"f"(bb));
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
                asm volatile("vse16.v v24, (%0)" ::"r"(xa + t) : "memory");
                asm volatile("vse16.v v26, (%0)" SPATZ_VST_FENCE ::"r"(xb + t) : "memory");
            }
        }
        i = i + gh == half ? 0 : i + gh;
    }

    return 0;
}
