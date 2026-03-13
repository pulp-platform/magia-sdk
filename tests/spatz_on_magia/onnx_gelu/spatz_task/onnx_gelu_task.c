#include "tile.h"
#include "onnx_gelu_params.h"

static inline void gelu_forward(const _Float16 *src, _Float16 *dst, size_t len)
{
    _Float16 C0 = 0.044715f;
    _Float16 C1 = 0.797884561f;     /* sqrt(2/pi) */
    _Float16 TANH_MIN = -5.0f;
    _Float16 TANH_MAX = 5.0f;
    _Float16 BIAS = 15360.0f;
    _Float16 COEF = 1486.0f;
    _Float16 HALF = 0.5f;
    _Float16 ONE = 1.0f;
    _Float16 TWO = 2.0f;

    const _Float16 *p_src;
    _Float16 *p_dst;
    size_t avl;
    size_t vl;

    p_src = src;
    p_dst = dst;
    avl = len;

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
        asm volatile ("vle16.v v0, (%0)" :: "r"(p_src));
        asm volatile ("vfmul.vv v8, v0, v0");                   /* x^2              */
        asm volatile ("vfmul.vv v8, v8, v0");                   /* x^3              */
        asm volatile ("vfmul.vf v8, v8, %0" :: "f"(C0));        /* 0.044715 * x^3   */
        asm volatile ("vfadd.vv v8, v8, v0");                   /* x + 0.044715 x^3 */
        asm volatile ("vfmul.vf v8, v8, %0" :: "f"(C1));        /* sqrt(2/pi)*(...) */

        /* ---------- fast tanh approximation ---------- */

        /* clamp for stability */
        asm volatile ("vfmin.vf v8, v8, %0" :: "f"(TANH_MAX));
        asm volatile ("vfmax.vf v8, v8, %0" :: "f"(TANH_MIN));

        asm volatile ("vfmul.vf v16, v8, %0" :: "f"(TWO));
        asm volatile ("vfmul.vf v16, v16, %0" :: "f"(COEF));
        asm volatile ("vfadd.vf v16, v16, %0" :: "f"(BIAS));
        asm volatile ("vfcvt.rtz.xu.f.v v16, v16");

        asm volatile ("vfadd.vf v8, v16, %0" :: "f"(ONE));
        asm volatile ("vfsub.vf v16, v16, %0" :: "f"(ONE));
        asm volatile ("vfdiv.vv v8, v16, v8");

        /* --------------------------------------------- */

        asm volatile ("vfadd.vf v8, v8, %0" :: "f"(ONE));       /* 1 + tanh         */
        asm volatile ("vfmul.vv v8, v8, v0");                   /* x * (...)        */
        asm volatile ("vfmul.vf v8, v8, %0" :: "f"(HALF));      /* 0.5 * (...)      */

        asm volatile ("vse16.v v8, (%0)" :: "r"(p_dst));

        p_src += vl;
        p_dst += vl;
    }
}

int onnx_gelu_task(void)
{
    volatile onnx_gelu_params_t *params;
    uintptr_t params_addr;
    _Float16 *src;
    _Float16 *dst;
    size_t len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile onnx_gelu_params_t *) params_addr;

    src = (_Float16 *)params->addr_src;
    dst = (_Float16 *)params->addr_res;
    len = params->len;

    gelu_forward(src, dst, len);

    return 0;
}
