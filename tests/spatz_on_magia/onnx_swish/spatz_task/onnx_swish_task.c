#include "tile.h"
#include "onnx_swish_params.h"

static inline void swish(const _Float16 *src, _Float16 *dst, const _Float16 alpha, const size_t len)
{
    register _Float16 BIAS asm ("fs0") = 15360.0f;
    register _Float16 COEF asm ("fs1") = 1477.0f;
    register _Float16 ONE  asm ("fs2") = 1.0f;
    const _Float16 *p_src;
    _Float16 *p_dst;
    size_t avl;
    size_t vl;

    p_src = src;
    p_dst = dst;
    avl = len;

    asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile ("vfmv.v.f v8, %0" :: "f"(ONE));
    asm volatile ("vfmv.v.f v16, %0" :: "f"(alpha));

    for(; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v0,( %0)" :: "r"(p_src));

        asm volatile ("vfmul.vv v24, v0, v16");
        asm volatile ("vfsgnjn.vv v24, v24, v24");

        /* ---------- fast exp approximation ---------- */
        asm volatile ("vfmul.vf v24, v24, %0" :: "f"(COEF));
        asm volatile ("vfadd.vf v24, v24, %0" :: "f"(BIAS));
        asm volatile ("vfcvt.rtz.xu.f.v v24, v24");
        /* -------------------------------------------- */

        asm volatile ("vfadd.vf v24, v24, %0" :: "f"(ONE));
        asm volatile ("vfdiv.vv v0, v0, v24");


        asm volatile ("vse16.v v0, (%0)" :: "r"(p_dst));

        p_src += vl;
        p_dst += vl;
    }
}

int onnx_swish_task(void)
{
    volatile onnx_swish_params_t *params;
    uintptr_t params_addr;
    _Float16 alpha;
    _Float16 *X;
    _Float16 *Y;
    size_t len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_swish_params_t *) params_addr;

    alpha = *(_Float16 *)params->alpha;
    X = (_Float16 *)params->chunk_X;
    Y = (_Float16 *)params->chunk_Y;
    len = params->len;

    swish(X, Y, alpha, len);

    return 0;
}
