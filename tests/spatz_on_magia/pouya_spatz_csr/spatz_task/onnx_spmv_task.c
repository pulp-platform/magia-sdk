#include "tile.h"
#include "onnx_spmv_params.h"

/*
 * SpMV (CSR format): Y = A * x
 *
 * Vectorization axis: nonzeros WITHIN a row, not rows themselves - row
 * lengths are irregular, so vectorizing across M would leave lanes idle
 * whenever a row is short. Each row's dot product A[i,:] . x is accumulated
 * lane-wise into a vector accumulator, then reduced to a scalar Y[i].
 *
 * col_idx is uint32_t (wide range -> doesn't limit matrix dimensions), but
 * values/x/Y stay fp16 and the FMA itself runs entirely at e16,m1 - exactly
 * like the GEMV kernel's fp16 accumulation. We do NOT widen values/x up to
 * fp32 to "match" the 32-bit index: widening always doubles the source
 * LMUL, which would force a fractional (mf2) source group to land on an m1
 * destination - the exact fractional-LMUL situation we're avoiding.
 * Instead we exploit RVV's mixed-width indexed load: vloxei32.v can gather
 * 16-bit data elements through 32-bit byte offsets. col_idx only ever
 * occupies an m2 register group as a side effect of that width mismatch
 * (EMUL = EEW/SEW * LMUL = 32/16 * 1 = 2), never as a separate choice.
 *
 * Registers: v8 = values, v10 = gathered x, v12 = row accumulator (persists
 * across chunks within a row), v14 = reduction identity, v15 = reduction
 * result, v16:v17 = col_idx (m2). All disjoint, no group overlap.
 */

static void spmv(const uint32_t *row_ptr,
                 const uint32_t *col_idx,
                 const _Float16 *values,
                 const _Float16 *x,
                 _Float16 *Y,
                 const size_t dim_M)
{
    register _Float16 ZERO asm("fs0") = 0.0f;
    uint32_t row_start;
    uint32_t row_end;
    uint32_t nnz;
    uint32_t j;
    size_t max_vl;
    size_t vl;
    size_t remaining;

    for (uint32_t i = 0; i < dim_M; i++) {
        row_start = row_ptr[i];
        row_end   = row_ptr[i + 1];
        nnz       = row_end - row_start;

        if (nnz == 0) {
            Y[i] = 0.0f;
            continue;
        }

        asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(max_vl) : "r"((size_t)nnz));
        asm volatile("vfmv.v.f v12, %0" ::"f"(ZERO));

        j         = row_start;
        remaining = nnz;

        while (remaining > 0) {
            asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(remaining));

            /* Native e16 values; col_idx lands in an m2 group (EEW32/SEW16 = EMUL2).
             * col_idx entries are pre-shifted to BYTE offsets on the host side
             * (see init_data in main.c) - no runtime shift or vtype toggle here,
             * vtype stays at e16,m1 for the entire row/loop. */
            asm volatile("vle16.v v8, (%0)" ::"r"(values + j));
            asm volatile("vle32.v v16, (%0)" ::"r"(col_idx + j));

            asm volatile("vloxei32.v v10, (%0), v16" ::"r"(x));

            asm volatile("vfmacc.vv v12, v8, v10");

            j         += vl;
            remaining -= vl;
        }

        /* Reduce the row's max_vl partial sums to a scalar */
        asm volatile("vfmv.v.f v14, %0" ::"f"(ZERO));
        asm volatile("vfredusum.vs v15, v12, v14");

        register _Float16 result asm("fa0");
        asm volatile("vfmv.f.s %0, v15" : "=f"(result));
        Y[i] = result;
    }
}

int onnx_spmv_task(void)
{
    volatile onnx_spmv_params_t *params;
    uintptr_t params_addr;
    uint32_t *row_ptr;
    uint32_t *col_idx;
    _Float16 *values;
    _Float16 *x;
    _Float16 *Y;
    size_t M;

    params_addr = mmio32(SPATZ_DATA);
    params      = (volatile onnx_spmv_params_t *)params_addr;

    row_ptr = (uint32_t *)params->addr_row_ptr;
    col_idx = (uint32_t *)params->addr_col_idx;
    values  = (_Float16 *)params->addr_values;
    x       = (_Float16 *)params->addr_x;
    Y       = (_Float16 *)params->addr_Y;
    M       = params->M;

    spmv(row_ptr, col_idx, values, x, Y, M);

    return 0;
}