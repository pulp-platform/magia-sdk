#include "tile.h"
#include "sigmoid_fp16_spatz_params.h"

static inline void sigmoid(const _Float16 *src, _Float16 *dst, const size_t len)
{
    register _Float16 BIAS asm ("fs0") = 15360.0f;
    register _Float16 COEF asm ("fs1") = 1477.0f;
    register _Float16 ZERO asm ("fs2") = 0.0f;
    register _Float16 ONE  asm ("fs3") = 1.0f;
    /*
     * The fast exp below reinterprets COEF*(-x) + BIAS as FP16 bits, so it is only usable
     * while that value stays a positive finite FP16: past -x = 11.09 it crosses 32768 and
     * the "exp" comes back with its sign bit set, and past -x = 34 the vfcvt saturates to
     * 0xFFFF, which *is* an FP16 NaN. Clamping x at -11 caps the result at sigmoid(-11) =
     * 1.7e-5, i.e. an absolute error below 1.5e-5 - two orders under the fast exp's own
     * ~0.01 deviation from a true sigmoid.
     */
    register _Float16 MIN  asm ("fs4") = -11.0f;
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

        asm volatile ("vfmax.vf v0, v0, %0" :: "f"(MIN));

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

int sigmoid_fp16_spatz_task(void)
{
    volatile sigmoid_fp16_spatz_params_t *params;
    uintptr_t params_addr;
    _Float16 *X;
    _Float16 *Y;
    size_t len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile sigmoid_fp16_spatz_params_t *) params_addr;

    X = (_Float16 *) params->shard_X;
    Y = (_Float16 *) params->shard_Y;
    len = params->len;

    sigmoid(X, Y, len);
}
