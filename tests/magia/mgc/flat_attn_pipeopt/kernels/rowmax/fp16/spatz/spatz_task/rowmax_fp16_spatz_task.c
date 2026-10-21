#include "tile.h"
#include "rowmax_fp16_spatz_params.h"

/* Unit-stride vector accesses must be 4-byte aligned. */
#define MISALIGNED(p)  (((uintptr_t)(p)) & 0x2)

/* Any rows: reducemax over the last axis with one lane per row (column-strided loads), and
   the element-wise max with the previous row max. Cold: linked out of the hot tasks. */
static __attribute__((noinline, cold)) void rowmax_strided(
    const _Float16 *X, const _Float16 *P, _Float16 *M, size_t rows, size_t cols)
{
    size_t row_stride = cols * sizeof(_Float16);
    size_t vl;

    for (size_t r = 0; r < rows; r += vl) {
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(rows - r));

        asm volatile("vlse16.v v0, (%0), %1" ::"r"(X + r * cols), "r"(row_stride));
        for (size_t c = 1; c < cols; c++) {
            asm volatile("vlse16.v v8, (%0), %1" ::"r"(X + r * cols + c), "r"(row_stride));
            asm volatile("vfmax.vv v0, v0, v8");
        }

        if (P) {
            asm volatile("vle16.v v8, (%0)" ::"r"(P + r));
            asm volatile("vfmax.vv v0, v0, v8");
        }

        asm volatile("vse16.v v0, (%0)" ::"r"(M + r) : "memory");
    }
}

/* Fuses reducemax over the last axis and the element-wise max with the previous row max.
   Rows are read with unit-stride loads and reduced with vfredmax: a column-strided load hits a
   single L1 bank when the row pitch is a multiple of 128 B. Each row max is stored by a
   1-element vector store, so the Snitch never waits for it. */
int rowmax_fp16_spatz_task(void)
{
    const rowmax_fp16_spatz_params_t *params;
    const _Float16 *X;
    const _Float16 *P;
    _Float16 *M;
    size_t rows;
    size_t cols;
    size_t vl, vl0;

    params = (const rowmax_fp16_spatz_params_t *)mmio32(SPATZ_DATA);

    X    = (const _Float16 *)params->shard_X;
    P    = (const _Float16 *)params->shard_P;
    M    = (_Float16 *)params->shard_M;
    rows = params->dims & 0xffff;
    cols = params->dims >> 16;

    if (MISALIGNED(M) || MISALIGNED(P))
        return 1;
    if (rows == 0 || cols == 0)
        return 0;
    if (MISALIGNED(X) || (cols & 1)) {
        rowmax_strided(X, P, M, rows, cols);
        return 0;
    }

    for (size_t r = 0; r < rows; r++, X += cols) {
        const _Float16 *x = X;
        size_t avl;

        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl0) : "r"(cols));
        asm volatile("vle16.v v0, (%0)" ::"r"(x));
        for (x += vl0, avl = cols - vl0; avl > 0; avl -= vl, x += vl) {
            asm volatile("vsetvli %0, %1, e16, m8, tu, ma" : "=r"(vl) : "r"(avl));
            asm volatile("vle16.v v8, (%0)" ::"r"(x));
            asm volatile("vfmax.vv v0, v0, v8");
        }
        asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(vl0));
        asm volatile("vfredmax.vs v16, v0, v0");
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(1));
        asm volatile("vse16.v v16, (%0)" ::"r"(M + r) : "memory");
    }

    if (P) {
        _Float16 *m = M;
        for (size_t avl = rows; avl > 0; avl -= vl, m += vl, P += vl) {
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(avl));
            asm volatile("vle16.v v0, (%0)" ::"r"(m));
            asm volatile("vle16.v v8, (%0)" ::"r"(P));
            asm volatile("vfmax.vv v0, v0, v8");
            asm volatile("vse16.v v0, (%0)" ::"r"(m) : "memory");
        }
    }

    return 0;
}
