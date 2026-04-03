#include "tile.h"
#include "onnx_sigmoid_params.h"

static inline void sigmoid(const _Float16 *src, _Float16 *dst, const size_t len)
{
    register _Float16 BIAS asm ("fs0") = 15360.0f;
    register _Float16 COEF asm ("fs1") = 1477.0f;
    register _Float16 ZERO asm ("fs2") = 0.0f;
    register _Float16 ONE  asm ("fs3") = 1.0f;
    register _Float16 MIN  asm ("fs4") = -5.0f;
    const _Float16 *p_src;
    _Float16 *p_dst;
    size_t avl;
    size_t vl;

    p_src = src;
    p_dst = dst;
    avl = len;

    asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
    asm volatile ("vfmv.v.f v8, %0" :: "f"(ONE));

    for(; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile ("vle16.v v0,( %0)" :: "r"(p_src));

        asm volatile ("vfsgnjn.vv v0, v0, v0");

        /* ---------- fast exp approximation ---------- */
        asm volatile ("vfmul.vf v0, v0, %0" :: "f"(COEF));
        asm volatile ("vfadd.vf v0, v0, %0" :: "f"(BIAS));
        asm volatile ("vfcvt.rtz.xu.f.v v0, v0");
        /* -------------------------------------------- */

        asm volatile ("vfadd.vf v0, v0, %0" :: "f"(ONE));
        asm volatile ("vfdiv.vv v0, v8, v0");

        asm volatile ("vse16.v v0, (%0)" :: "r"(p_dst));

        p_src += vl;
        p_dst += vl;
    }
}

int onnx_sigmoid_task(void)
{
    volatile onnx_sigmoid_params_t *params;
    uintptr_t params_addr;
    _Float16 *src;
    _Float16 *dst;
    size_t len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_sigmoid_params_t *) params_addr;

    src = (_Float16 *)params->addr_input;
    dst = (_Float16 *)params->addr_res;
    len = params->len;

    sigmoid(src, dst, len);

    return 0;
}
