#include "tile.h"
#include "tanh_fp16_spatz_params.h"


static inline void fast_tanh(const _Float16 *src, _Float16 *dst, size_t len)
{
    _Float16 TANH_MIN = -5.0f;
    _Float16 TANH_MAX = 5.0f;
    _Float16 BIAS = 15296.0f;
    _Float16 COEF = 1486.0f;
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

        /* clamp for stability */
        asm volatile ("vfmin.vf v0, v0, %0" :: "f"(TANH_MAX));
        asm volatile ("vfmax.vf v0, v0, %0" :: "f"(TANH_MIN));

        asm volatile ("vfmul.vf v16, v0, %0" :: "f"(TWO));
        asm volatile ("vfmul.vf v16, v16, %0" :: "f"(COEF));
        asm volatile ("vfadd.vf v16, v16, %0" :: "f"(BIAS));
        asm volatile ("vfcvt.rtz.xu.f.v v16, v16");

        asm volatile ("vfadd.vf v8, v16, %0" :: "f"(ONE));
        asm volatile ("vfsub.vf v16, v16, %0" :: "f"(ONE));
        asm volatile ("vfdiv.vv v8, v16, v8");

        asm volatile ("vse16.v v8, (%0)" :: "r"(p_dst));

        p_src += vl;
        p_dst += vl;
    }
}

int tanh_fp16_spatz_task(void)
{
    volatile tanh_fp16_spatz_params_t *params;
    uintptr_t params_addr;
    _Float16 *output;
    _Float16 *input;
    size_t len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile tanh_fp16_spatz_params_t *) params_addr;

    input = (_Float16 *)params->shard_input;
    output = (_Float16 *)params->shard_output;
    len = params->len;

    fast_tanh(input, output, len);

    return 0;
}
