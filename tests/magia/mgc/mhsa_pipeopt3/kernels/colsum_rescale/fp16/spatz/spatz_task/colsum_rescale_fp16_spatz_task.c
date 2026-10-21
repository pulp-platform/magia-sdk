#include "tile.h"
#include "colsum_rescale_fp16_spatz_params.h"

/* Unit-stride vector accesses must be 4-byte aligned. */
#define MISALIGNED(p)  (((uintptr_t)(p)) & 0x2)

/* Fuses reducesum over the first axis (fp32 accumulation, seeded with S as row 0, as
   reducesum), mul and add, with the same arithmetic as the separate kernels. */
int colsum_rescale_fp16_spatz_task(void)
{
    const colsum_rescale_fp16_spatz_params_t *params;
    register _Float16 zero asm("fs0") = 0.0f;
    _Float16 *S;
    const _Float16 *G;
    _Float16 *P;
    const _Float16 *A;
    size_t rows;
    size_t cols;
    size_t vl;

    params = (const colsum_rescale_fp16_spatz_params_t *)mmio32(SPATZ_DATA);

    S    = (_Float16 *)params->shard_S;
    G    = (const _Float16 *)params->shard_G;
    P    = (_Float16 *)params->shard_P;
    A    = (const _Float16 *)params->shard_A;
    rows = params->dims & 0xffff;
    cols = params->dims >> 16;

    if (MISALIGNED(S) || MISALIGNED(G) || MISALIGNED(P) || MISALIGNED(A) || (cols & 1))
        return 1;

    for (size_t c = 0; c < cols; c += vl) {
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(cols - c));

        asm volatile("vle16.v v8, (%0)" ::"r"(S + c));
        asm volatile("vfwadd.vf v0, v8, %0" ::"f"(zero));
        for (size_t r = 1; r < rows; r++) {
            asm volatile("vle16.v v8, (%0)" ::"r"(G + r * cols + c));
            asm volatile("vfwadd.wv v0, v0, v8");
        }
        asm volatile("vfncvt.f.f.w v8, v0");

        if (P) {
            asm volatile("vle16.v v16, (%0)" ::"r"(A + c));
            asm volatile("vle16.v v24, (%0)" ::"r"(P + c));
            asm volatile("vfmul.vv v16, v16, v24");
            asm volatile("vse16.v v16, (%0)" ::"r"(P + c) : "memory");
            asm volatile("vfadd.vv v8, v16, v8");
        }

        asm volatile("vse16.v v8, (%0)" ::"r"(S + c) : "memory");
    }

    return 0;
}
