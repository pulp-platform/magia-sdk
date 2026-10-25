#include "tile.h"
#include "bias_rope_t_fp16_spatz_params.h"

/* bias_rope on a transposed block (channels along the rows, tokens along the columns, as Kt):
   X[r, t] += bias[r] (fp16 adds), then rows (2i, 2i + 1) of a head rotated by angle i,
     y[2i, t]     = x[2i, t] * cos[i, t] - x[2i + 1, t] * sin[i, t]
     y[2i + 1, t] = x[2i + 1, t] * cos[i, t] + x[2i, t] * sin[i, t]
   with the same arithmetic as bias_rope (exact fp32 products, fp32 then fp16 rounding), all
   unit-stride along the tokens. The block may start inside a head (row0) and straddle heads. */
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
    size_t f;
    size_t vl;

    params = (const bias_rope_t_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    X      = (_Float16 *)params->shard_X;
    B      = (const _Float16 *)params->bias;
    C      = (const _Float16 *)params->cos;
    S      = (const _Float16 *)params->sin;
    rows   = params->dims & 0xffff;
    cols   = params->dims >> 16;
    half   = params->head_dim / 2;
    f      = params->row0 / 2;

    for (size_t i = 0; i + 1 < rows; i += 2) {
        _Float16 *xa       = X + i * cols;
        _Float16 *xb       = xa + cols;
        const _Float16 *cf = C + f * cols;
        const _Float16 *sf = S + f * cols;
        _Float16 ba        = B[i];
        _Float16 bb        = B[i + 1];

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
        f = f + 1 == half ? 0 : f + 1;
    }

    return 0;
}
