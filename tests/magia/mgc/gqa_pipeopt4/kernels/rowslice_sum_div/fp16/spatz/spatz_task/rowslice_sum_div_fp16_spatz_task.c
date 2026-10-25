#include "tile.h"
#include "rowslice_sum_div_fp16_spatz_params.h"
#include "utils/spatz_recip.h"

/* Fuses the reduce-scatter sum of the n partial outputs and the row division (add, then rowdiv,
   with the same arithmetic as the separate kernels: fp16 adds, division as a multiplication by
   1/s with one correction step). Each row's divisor is loaded before the previous row's last
   store: a scalar load stalls on a pending vector store, which would drain the VU at every row. */
int rowslice_sum_div_fp16_spatz_task(void)
{
    const rowslice_sum_div_fp16_spatz_params_t *params;
    _Float16 *Y;
    const _Float16 *O;
    const _Float16 *G;
    const _Float16 *S;
    size_t rows;
    size_t cols;
    size_t n;
    size_t own;
    size_t vl;
    _Float16 s;
    _Float16 s_inv;
    _Float16 next;

    params = (const rowslice_sum_div_fp16_spatz_params_t *)mmio32(SPATZ_DATA);

    Y    = (_Float16 *)params->shard_Y;
    O    = (const _Float16 *)params->shard_O;
    G    = (const _Float16 *)params->shard_G;
    S    = (const _Float16 *)params->shard_S;
    rows = params->dims & 0xffff;
    cols = params->dims >> 16;
    n    = params->slices & 0xffff;
    own  = params->slices >> 16;

    if (rows == 0)
        return 0;

    s = S[0];
    for (size_t r = 0; r < rows; r++) {
        const _Float16 *s_next = S + (r + 1 < rows ? r + 1 : r);

        s_inv = spatz_recip_f16(s);
        for (size_t c = 0; c < cols; c += vl) {
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(cols - c));

            asm volatile("vle16.v v0, (%0)" ::"r"(O + r * cols + c));
            for (size_t k = 0; k < n; k++) {
                if (k == own)
                    continue;
                asm volatile("vle16.v v8, (%0)" ::"r"(G + (k * rows + r) * cols + c));
                asm volatile("vfadd.vv v0, v0, v8");
            }

            SPATZ_VFDIV_VF("v8", "v0", "v16", s, s_inv);

            asm volatile("flh %0, 0(%1)" : "=f"(next) : "r"(s_next));

            asm volatile("vse16.v v8, (%0)" SPATZ_VST_FENCE ::"r"(Y + r * cols + c) : "memory");
        }
        s = next;
    }

    return 0;
}
