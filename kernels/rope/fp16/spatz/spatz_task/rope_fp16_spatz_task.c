#include "tile.h"
#include "rope_fp16_spatz_params.h"

/* The four products are exact in fp32 (vfwmul, fp16 x fp16), the sum and difference are rounded
   once in fp32 and once to fp16. */
int rope_fp16_spatz_task(void)
{
    volatile rope_fp16_spatz_params_t *params;
    const _Float16 *X;
    const _Float16 *C;
    const _Float16 *S;
    _Float16 *Y;
    size_t r_len;
    size_t n_heads;
    size_t half;

    params  = (volatile rope_fp16_spatz_params_t *)mmio32(SPATZ_DATA);
    X       = (const _Float16 *)params->shard_X;
    Y       = (_Float16 *)params->shard_Y;
    C       = (const _Float16 *)params->cos;
    S       = (const _Float16 *)params->sin;
    r_len   = params->r_len;
    n_heads = params->n_heads;
    half    = params->head_dim / 2;

    for (size_t r = 0; r < r_len; r++) {
        for (size_t h = 0; h < n_heads; h++) {
            const _Float16 *x1 = X + (r * n_heads + h) * 2 * half;
            const _Float16 *x2 = x1 + half;
            const _Float16 *c  = C + r * half;
            const _Float16 *s  = S + r * half;
            _Float16 *y1       = Y + (r * n_heads + h) * 2 * half;
            _Float16 *y2       = y1 + half;
            size_t avl;
            size_t vl;

            for (avl = half; avl > 0; avl -= vl) {
                asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(vl) : "r"(avl));
                asm volatile("vle16.v v0, (%0)" ::"r"(x1));
                asm volatile("vle16.v v2, (%0)" ::"r"(x2));
                asm volatile("vle16.v v4, (%0)" ::"r"(c));
                asm volatile("vle16.v v6, (%0)" ::"r"(s));
                asm volatile("vfwmul.vv v8, v0, v4");  /* x1 * cos */
                asm volatile("vfwmul.vv v12, v2, v6"); /* x2 * sin */
                asm volatile("vfwmul.vv v16, v2, v4"); /* x2 * cos */
                asm volatile("vfwmul.vv v20, v0, v6"); /* x1 * sin */

                asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(vl));
                asm volatile("vfsub.vv v8, v8, v12");
                asm volatile("vfadd.vv v16, v16, v20");

                asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(vl));
                asm volatile("vfncvt.f.f.w v24, v8");
                asm volatile("vfncvt.f.f.w v26, v16");
                asm volatile("vse16.v v24, (%0)" ::"r"(y1) : "memory");
                asm volatile("vse16.v v26, (%0)" ::"r"(y2) : "memory");

                x1 += vl;
                x2 += vl;
                c += vl;
                s += vl;
                y1 += vl;
                y2 += vl;
            }
        }
    }

    return 0;
}
