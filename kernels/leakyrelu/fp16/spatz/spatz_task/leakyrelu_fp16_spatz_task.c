#include "tile.h"
#include "leakyrelu_fp16_spatz_params.h"

static void leakyrelu(const _Float16 *src, _Float16 *dst, const _Float16 alpha, const size_t len)
{
    register _Float16 ZERO asm ("fs1") = 0.0f;
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

        asm volatile ("vmflt.vf v0, v8, %0" :: "f"(ZERO));
        asm volatile ("vfmul.vf v8, v8, %0, v0.t" :: "f"(alpha));

        asm volatile ("vse16.v v8, (%0)" :: "r"(p_dst));

        p_src += vl;
        p_dst += vl;
    }
}

int leakyrelu_fp16_spatz_task(void)
{
    volatile leakyrelu_fp16_spatz_params_t *params;
    uintptr_t params_addr;

    _Float16 alpha;
    _Float16 *X;
    _Float16 *Y;
    size_t len;

    params_addr = mmio32(SPATZ_DATA);
    params = (volatile leakyrelu_fp16_spatz_params_t *) params_addr;

    alpha = *(_Float16 *) params->alpha;
    X = (_Float16 *)params->shard_X;
    Y = (_Float16 *)params->shard_Y;
    len = params->len;

    leakyrelu(X, Y, alpha, len);

    return 0;
}
