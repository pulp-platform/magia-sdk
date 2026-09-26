#ifndef SPATZ_CSR_PARAMS_H_
#define SPATZ_CSR_PARAMS_H_

#include <stdint.h>

/*
 * Parameter block passed from the CV32 host to the Spatz task through
 * SPATZ_DATA. Every field is a 32-bit value (uintptr_t or uint32_t) so
 * host and task agree on layout without any padding/alignment surprises.
 *
 * All *_addr fields are absolute L1 addresses (this tile's L1, see
 * mem_layout.h) pointing at:
 *   addr_row_ptr : uint32_t[M+1]   CSR row pointer
 *   addr_col_idx : uint16_t[nnz]   CSR column indices, PRE-SCALED to byte
 *                                  offsets (col * sizeof(float16)) by the
 *                                  host, ready for vloxei16.v
 *   addr_values  : float16[nnz]    CSR nonzero values
 *   addr_x       : float16[K]      dense input vector
 *   addr_Y       : float16[M]      dense output vector (written by task)
 *   addr_G       : float16[M]      golden reference (host compares Y vs G,
 *                                  kept in L1 for debugging only)
 */
typedef struct {
    uintptr_t addr_row_ptr;
    uintptr_t addr_col_idx;
    uintptr_t addr_values;
    uintptr_t addr_x;
    uintptr_t addr_Y;
    uintptr_t addr_G;
    uint32_t  M;    /* number of rows of A / length of Y        */
    uint32_t  K;    /* number of cols of A / length of x        */
    uint32_t  nnz;  /* number of nonzeros (row_ptr[M])           */
} spatz_csr_params_t;

#endif /* SPATZ_CSR_PARAMS_H_ */