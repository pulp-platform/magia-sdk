#include "tile.h"
#include "onnx_spmv_params.h"

/*
 * SPATZ-vectorized inner loop for test.c's SA-tiled, address-resolved CSR
 * format.
 *
 * Departure from a textbook indexed-gather SpMV kernel: each CSR entry is
 * two 32-bit words, {value, x_addr}, where x_addr is an already-resolved
 * ABSOLUTE L1 byte address (possibly on a remote core, via the same
 * transparent NoC access the scalar version relies on when it does
 * `*(volatile int16_t *)x_addr`) -- not an offset from one shared base.
 * There is therefore no single base register to feed an indexed load, so
 * we gather with base = 0 and let each element of the index vector BE the
 * absolute address: effective_addr = 0 + index = index.
 *
 * ASSUMPTION (flag this if wrong): `value`, though stored as a 32-bit word
 * for uniform 8-byte entry alignment with the address, fits in 16 bits.
 * We load only its low 16 bits (little-endian) and use widening
 * vwmacc.vv (i16*i16 -> i32) so every register group stays at integer
 * LMUL -- same reasoning the fp16 reference kernel uses to avoid a
 * fractional-LMUL widen. If values need the full 32 bits, this needs to
 * become a vsext.vf2-based path instead (tell me and I'll redo it).
 *
 * IMPORTANT CAVEAT: this assumes the vector load/store unit can issue
 * vloxei32 gathers to REMOTE L1 addresses transparently, the same way the
 * scalar core's `*(volatile int16_t *)x_addr` does in test.c. If SPATZ's
 * vector memory port does NOT have that NoC transparency (only the
 * scalar/DMA paths do), this kernel is not directly usable as-is and the
 * remote-owned nonzeros would need to stay on the scalar path (or be
 * DMA'd into a local scratch first) -- that's a hardware capability
 * question I can't resolve without your platform's memory-map docs.
 *
 * Registers: v8 = value_lo16 (m1), v16:v17 = x_addr (m2, EEW32/SEW16),
 * v10 = gathered x (m1), v12:v13 = i32 accumulator (m2, persists across
 * chunks within a row), v14:v15 = reduction identity/result (m2/m1).
 */

static void csr_tile_spmv(const uint32_t *local_rowptr,
                           const uint32_t *valcol_buf, /* interleaved {value, x_addr} */
                           int32_t *local_y,
                           uint32_t tile_row_start,
                           uint32_t tile_row_end,
                           uint32_t start_nnz)
{
    uint32_t local_start, local_end, nnz, j, remaining;
    size_t max_vl, vl;

    for (uint32_t i = tile_row_start; i < tile_row_end; i++) {
        local_start = local_rowptr[i]     - start_nnz;
        local_end   = local_rowptr[i + 1] - start_nnz;
        nnz         = local_end - local_start;

        if (nnz == 0) {
            local_y[i] = 0;
            continue;
        }

        /* max_vl in e16,m1 terms == VLMAX in e32,m2 terms (LMUL doubles as
         * SEW doubles), so the same count zeroes exactly the accumulator
         * lanes this row's chunks will touch. */
        asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(max_vl) : "r"((size_t)nnz));
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" :: "r"(max_vl));
        asm volatile("vmv.v.x v12, zero");

        j         = local_start;
        remaining = nnz;

        while (remaining > 0) {
            asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(remaining));

            /* value low 16 bits, strided 8 bytes (skip the addr word) */
            asm volatile("vlse16.v v8, (%0), %1"
                         :: "r"((const int16_t *)&valcol_buf[2 * j]), "r"(8));

            /* x_addr (absolute), strided 8 bytes, +4 byte offset */
            asm volatile("vlse32.v v16, (%0), %1"
                         :: "r"((const uint8_t *)&valcol_buf[2 * j] + 4), "r"(8));

            /* gather int16 x values directly from the absolute addresses
             * in v16 -- base=0 so offset IS the address */
            asm volatile("vloxei32.v v10, (%0), v16" :: "r"(0));

            /* widening MAC: v12 += v8 * v10  (i16 x i16 -> i32) */
            asm volatile("vwmacc.vv v12, v8, v10");

            j         += vl;
            remaining -= vl;
        }

        /* Reduce this row's i32 accumulator to a scalar */
        asm volatile("vsetvli zero, %0, e32, m2, ta, ma" :: "r"(max_vl));
        asm volatile("vmv.v.x v14, zero");
        asm volatile("vredsum.vs v15, v12, v14");

        register int32_t result asm("a0");
        asm volatile("vmv.x.s %0, v15" : "=r"(result));
        local_y[i] = result;
    }
}

int csr_tile_spmv_task(void)
{
    volatile csr_tile_params_t *params;
    uintptr_t params_addr;
    const uint32_t *local_rowptr;
    const uint32_t *valcol_buf;
    int32_t *local_y;

    params_addr = mmio32(SPATZ_DATA);
    params      = (volatile csr_tile_params_t *)params_addr;

    local_rowptr = (const uint32_t *)params->addr_local_rowptr;
    valcol_buf   = (const uint32_t *)params->addr_valcol_buf;
    local_y      = (int32_t *)params->addr_local_y;

    csr_tile_spmv(local_rowptr, valcol_buf, local_y,
                  params->tile_row_start, params->tile_row_end, params->start_nnz);

    return 0;
}
