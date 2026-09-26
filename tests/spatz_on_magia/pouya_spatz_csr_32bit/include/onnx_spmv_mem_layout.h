#ifndef SPATZ_CSR_MEM_LAYOUT_H_
#define SPATZ_CSR_MEM_LAYOUT_H_

#include "tile.h"        /* L1_BASE, L1_TILE_OFFSET, get_hartid()          */
#include "onnx_spmv_params.h"
#include "data.h"        /* DIM_M, DIM_K, NNZ                              */

/* Round a size/address up to the next multiple of 4 bytes. */
#define ALIGN_4B(a)   (((a) + 3u) & ~3u)

/* This tile's L1 base (own tile -> get_hartid()). */
#define L1_BASE_TILE  (L1_BASE + get_hartid() * L1_TILE_OFFSET)

/* ------------------------------------------------------------------ *
 * Consecutive, 4B-aligned layout inside this tile's L1:
 *   [ params_t | row_ptr | col_idx | values | x | Y | G ]
 *
 * fp32 version: every buffer is natively 4-byte-wide (row_ptr: uint32_t,
 * col_idx: uint32_t BYTE offsets, values/x/Y/G: float), so every element
 * is individually 4B aligned as long as each buffer's base is.
 * ------------------------------------------------------------------ */

#define SPATZ_CSR_PARAMS_BASE   L1_BASE_TILE
#define SPATZ_CSR_PARAMS_SIZE   ALIGN_4B(sizeof(spatz_csr_params_t))

#define ROW_PTR_BASE  ALIGN_4B(SPATZ_CSR_PARAMS_BASE + SPATZ_CSR_PARAMS_SIZE)
#define ROW_PTR_SIZE  ALIGN_4B((DIM_M + 1u) * sizeof(uint32_t))

#define COL_IDX_BASE  ALIGN_4B(ROW_PTR_BASE + ROW_PTR_SIZE)
#define COL_IDX_SIZE  ALIGN_4B(NNZ * sizeof(uint32_t))   /* byte offsets, 32b each */

#define VALUES_BASE   ALIGN_4B(COL_IDX_BASE + COL_IDX_SIZE)
#define VALUES_SIZE   ALIGN_4B(NNZ * sizeof(float))

#define X_BASE        ALIGN_4B(VALUES_BASE + VALUES_SIZE)
#define X_SIZE        ALIGN_4B(DIM_K * sizeof(float))

#define Y_BASE        ALIGN_4B(X_BASE + X_SIZE)
#define Y_SIZE        ALIGN_4B(DIM_M * sizeof(float))

#define G_BASE        ALIGN_4B(Y_BASE + Y_SIZE)
#define G_SIZE        ALIGN_4B(DIM_M * sizeof(float))

#define SPATZ_CSR_END_ADDR   (G_BASE + G_SIZE)

/* TODO [per guide, section 6/12]: L1 size per tile / L1_TILE_OFFSET are not
 * yet confirmed from the MAGIA repo. Once known, add:
 *   _Static_assert(SPATZ_CSR_END_ADDR - L1_BASE_TILE <= L1_SIZE,
 *                  "spatz_csr layout overflows tile L1");
 */

#endif /* SPATZ_CSR_MEM_LAYOUT_H_ */