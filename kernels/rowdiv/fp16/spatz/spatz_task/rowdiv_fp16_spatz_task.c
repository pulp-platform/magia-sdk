#include "tile.h"
#include "rowdiv_fp16_spatz_params.h"

/**
 * Divides one row by a scalar: dst[c] = src[c] / s.
 * Returns *s_next, loaded before the row's store: a scalar load stalls on a pending vector
 * store, which would drain the VU at every row.
 */
static inline _Float16 div_row(
    const _Float16 *src, _Float16 *dst, size_t len, _Float16 s, const _Float16 *s_next)
{
    const _Float16 *p_src;
    _Float16 *p_dst;
    _Float16 next;
    size_t avl;
    size_t vl;

    p_src = src;
    p_dst = dst;
    avl   = len;

    if (avl == 0)
        return *s_next;

    for (; avl > 0; avl -= vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));

        asm volatile("vle16.v v0, (%0)" ::"r"(p_src));

        asm volatile("vfdiv.vf v0, v0, %0" ::"f"(s));

        asm volatile("flh %0, 0(%1)" : "=f"(next) : "r"(s_next));

        asm volatile("vse16.v v0, (%0)" ::"r"(p_dst) : "memory");

        p_src += vl;
        p_dst += vl;
    }

    return next;
}

int rowdiv_fp16_spatz_task(void)
{
    volatile rowdiv_fp16_spatz_params_t *params;
    uintptr_t params_addr;
    const _Float16 *X;
    const _Float16 *S;
    _Float16 *Y;
    size_t cols;
    size_t rows;

    params_addr = mmio32(SPATZ_DATA);
    params      = (volatile rowdiv_fp16_spatz_params_t *)params_addr;

    X    = (const _Float16 *)params->shard_X;
    S    = (const _Float16 *)params->shard_S;
    Y    = (_Float16 *)params->shard_Y;
    cols = params->cols;
    rows = params->row_len;

    _Float16 s = rows ? S[0] : 0;
    for (size_t r = 0; r < rows; r++) {
        const _Float16 *s_next = S + (r + 1 < rows ? r + 1 : r);

        s = div_row(X + r * cols, Y + r * cols, cols, s, s_next);
    }

    return 0;
}
