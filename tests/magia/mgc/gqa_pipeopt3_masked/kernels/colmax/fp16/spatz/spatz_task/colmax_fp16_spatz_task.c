#include "tile.h"
#include "colmax_fp16_spatz_params.h"

/* Unit-stride vector accesses must be 4-byte aligned. */
#define MISALIGNED(p)  (((uintptr_t)(p)) & 0x2)

/* reducemax over the first axis for aligned rows: element-wise max of the rows, vectorized
   along the columns. */
int colmax_fp16_spatz_task(void)
{
    const colmax_fp16_spatz_params_t *params;
    const _Float16 *X;
    _Float16 *M;
    size_t rows;
    size_t cols;
    size_t vl;

    params = (const colmax_fp16_spatz_params_t *)mmio32(SPATZ_DATA);

    X    = (const _Float16 *)params->shard_X;
    M    = (_Float16 *)params->shard_M;
    rows = params->dims & 0xffff;
    cols = params->dims >> 16;

    if (MISALIGNED(X) || MISALIGNED(M) || (cols & 1))
        return 1;

    for (size_t c = 0; c < cols; c += vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(cols - c));

        asm volatile("vle16.v v0, (%0)" ::"r"(X + c));
        for (size_t r = 1; r < rows; r++) {
            asm volatile("vle16.v v8, (%0)" ::"r"(X + r * cols + c));
            asm volatile("vfmax.vv v0, v0, v8");
        }

        asm volatile("vse16.v v0, (%0)" ::"r"(M + c) : "memory");
    }

    return 0;
}
