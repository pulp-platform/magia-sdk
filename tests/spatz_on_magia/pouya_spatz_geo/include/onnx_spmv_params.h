#ifndef CSR_TILE_PARAMS_H_
#define CSR_TILE_PARAMS_H_

#include <stdint.h>

/*
 * Params block for the SPATZ-side vectorized inner loop of the SA-tiled,
 * address-resolved CSR SpMV (see test.c).
 *
 * NOTE: addr_local_rowptr / addr_valcol_buf / addr_local_y are L1 addresses
 * already local to THIS hart (test.c's local_rowptr / valcol_buf[current_buf]
 * / local_y). start_nnz is the same "start_nnz" test.c computes per tile,
 * needed to convert local_rowptr[i] (absolute nnz index) into an offset
 * within the current tile's valcol_buf.
 *
 * Adjust base-address macros below to match your actual
 * onnx_spmv_mem_layout.h / onnx_spmv_params.h conventions -- I don't have
 * those headers, so these are placeholders following the same pattern.
 */
typedef struct {
    uint32_t addr_local_rowptr; /* local_rowptr, L1                        */
    uint32_t addr_valcol_buf;   /* valcol_buf[current_buf], L1             */
    uint32_t addr_local_y;      /* local_y, L1                             */
    uint32_t tile_row_start;    /* local row index, inclusive              */
    uint32_t tile_row_end;      /* local row index, exclusive              */
    uint32_t start_nnz;         /* local_rowptr[tile_row_start] for this tile */
} csr_tile_params_t;

/*
 * There is deliberately no fixed CSR_TILE_PARAMS_BASE macro here: every
 * other L1 address in test.c (addr_x, addr_rowptr, addr_valcol_buf0/1,
 * addr_ylocal) is computed at RUNTIME by chaining off the previous
 * region's size, because local_x_count / local_rows / tile_buffer_bytes
 * are only known per-hart at runtime. A compile-time offset here could
 * silently overlap those regions. Instead, test.c computes
 * addr_csr_params the same way (chained after addr_ylocal) and passes it
 * directly to spatz_run_task_with_params -- see test_spatz.c.
 */

#define CSR_TILE_SPMV_TASK 1u /* matches FIRST_TASK_NAME id in CMakeLists */

#endif /* CSR_TILE_PARAMS_H_ */