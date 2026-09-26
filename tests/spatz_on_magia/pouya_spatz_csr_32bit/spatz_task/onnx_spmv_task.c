/*
 * spatz_csr_task.c  (fp32 version)
 *
 * See naming note in the fp16 version: rename/place as "spatz_csr_task.c"
 * to match the README's "<name>_task.c" convention and CMakeLists.txt.
 *
 * Kernel: CSR SpMV, Y = A * x, all data in fp32 (SPATZ_RVD=0 / ELEN=32,
 * so this is the native single-precision width -- no widening needed).
 *
 * Per row i (nnz_row = row_ptr[i+1] - row_ptr[i]):
 *   acc = 0
 *   for each chunk of up to VLMAX nonzeros in the row:
 *     v8  <- values[j .. j+vl)                     (vle32.v)
 *     v16 <- col_idx[j .. j+vl)  (BYTE offsets)     (vle32.v)
 *     v10 <- gather x at those byte offsets         (vloxei32.v)
 *     acc += v8 * v10                               (vfmacc.vv, tail-undisturbed)
 *   Y[i] = reduce_sum(acc)                          (vfredusum.vs + vfmv.f.s)
 *
 * Register map (e32/m1 throughout -> LMUL=1, EEW=SEW=32 for both the data
 * and the index load, so no fractional-LMUL index group like the fp16
 * version needed):
 *   v8  - values chunk
 *   v10 - gathered x chunk
 *   v12 - per-row accumulator (persists across chunks via "tu" policy)
 *   v14 - zero vector (reduction seed)
 *   v15 - reduction result
 *   v16 - col_idx chunk (byte offsets into x, now 32-bit)
 *
 * Known Issue 3 fix (guide, section 10) still applies: "tu, ma" in the
 * strip-mine loop, and vl re-fixed to max_vl before the reduction, so
 * rows longer than VLMAX (8 @ e32/m1/VLEN=256) don't lose accumulator
 * tail lanes.
 */

#include "tile.h"
#include "onnx_spmv_params.h"

int spatz_csr_task(void) {
    uintptr_t pa = mmio32(SPATZ_DATA);
    volatile spatz_csr_params_t *p = (volatile spatz_csr_params_t *)pa;

    const uint32_t *row_ptr = (const uint32_t *)p->addr_row_ptr;
    const uint32_t *col_idx = (const uint32_t *)p->addr_col_idx;  /* pre-scaled byte offsets */
    const float    *values  = (const float *)p->addr_values;
    const float    *x       = (const float *)p->addr_x;
    float          *Y       = (float *)p->addr_Y;
    const uint32_t  M       = p->M;

    register float ZERO asm("fs0") = 0.0f;

    for (uint32_t i = 0; i < M; i++) {
        uint32_t start  = row_ptr[i];
        uint32_t end    = row_ptr[i + 1];
        uint32_t nnz_row = end - start;

        if (nnz_row == 0) {
            Y[i] = 0.0f;
            continue;
        }

        const float    *val_row = values  + start;
        const uint32_t *idx_row = col_idx + start;

        uint32_t max_vl;
        asm volatile("vsetvli %0, %1, e32, m1, tu, ma"
                     : "=r"(max_vl) : "r"(nnz_row));

        asm volatile("vfmv.v.f v12, %0" :: "f"(ZERO));  /* acc = 0, full width */

        uint32_t remaining = nnz_row;
        uint32_t off       = 0;
        while (remaining > 0) {
            uint32_t vl;
            asm volatile("vsetvli %0, %1, e32, m1, tu, ma"
                         : "=r"(vl) : "r"(remaining));

            asm volatile("vle32.v v8,  (%0)" :: "r"(val_row + off));
            asm volatile("vle32.v v16, (%0)" :: "r"(idx_row + off));
            asm volatile("vloxei32.v v10, (%0), v16" :: "r"(x));
            asm volatile("vfmacc.vv v12, v8, v10");

            off       += vl;
            remaining -= vl;
        }

        asm volatile("vsetvli zero, %0, e32, m1, tu, ma" :: "r"(max_vl));
        asm volatile("vfmv.v.f v14, %0" :: "f"(ZERO));
        asm volatile("vfredusum.vs v15, v12, v14");

        float result;
        asm volatile("vfmv.f.s %0, v15" : "=f"(result));
        Y[i] = result;
    }

    return 0;
}