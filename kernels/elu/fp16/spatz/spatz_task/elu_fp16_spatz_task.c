#include "tile.h"
#include "elu_fp16_spatz_params.h"

static void elu(const _Float16 *src, _Float16 *dst, const _Float16 alpha, const size_t len)
{
    register _Float16 BIAS asm ("fs1") = 15360.0f;
    register _Float16 COEF asm ("fs2") = 1477.0f;
    register _Float16 ZERO asm ("fs3") = 0.0f;
    register _Float16 ONE  asm ("fs4") = 1.0f;
    register _Float16 MIN  asm ("fs5") = -5.0f;
    const _Float16 *p_src;
    _Float16 *p_dst;
    size_t avl;
    size_t vl;

    p_src = src;
    p_dst = dst;
    avl = len;

    for (; avl > 0; avl -= vl) {
        asm volatile ("vsetvli %0, %1, e16, m8, ta, mu" : "=r"(vl) : "r"(avl));
        asm volatile ("vle16.v v8, (%0)" :: "r"(p_src));

        /* mask negative values */
        // asm volatile ("vfmv.v.f v0, %0" :: "f"(ZERO));               /* TODO: is this needed? - ANSWER: probably not */
        asm volatile ("vmflt.vf v0, v8, %0" :: "f"(ZERO));

        /* ---------- fast exp approximation ---------- */
        /* clamp for stability */
        // asm volatile ("vfmax.vf v8, v8, %0, v0.t" :: "f"(MIN));      /* TODO: is this needed? - ANSWER: probably not */

        asm volatile ("vfmul.vf v8, v8, %0, v0.t" :: "f"(COEF));
        asm volatile ("vfadd.vf v8, v8, %0, v0.t" :: "f"(BIAS));
        asm volatile ("vfcvt.rtz.xu.f.v v8, v8, v0.t");
        /* -------------------------------------------- */

        asm volatile ("vfsub.vf v8, v8, %0, v0.t" :: "f"(ONE));
        asm volatile ("vfmul.vf v8, v8, %0, v0.t" :: "f"(alpha));

        asm volatile ("vse16.v v8, (%0)" :: "r"(p_dst));

        p_src += vl;
        p_dst += vl;
    }
}

int elu_fp16_spatz_task(void)
{
    volatile elu_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    _Float16 alpha;
    _Float16 *X;
    _Float16 *Y;
    size_t len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile elu_fp16_spatz_params_t *) params_addr;

    alpha = *(_Float16 *) params->alpha;
    X = (_Float16 *)params->shard_X;
    Y = (_Float16 *)params->shard_Y;
    len = params->len;

    elu(X, Y, alpha, len);

    return 0;
}
