#ifndef ONNX_SPMV_PARAMS_H_
#define ONNX_SPMV_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t addr_row_ptr; /* CSR row-pointer array (M+1 entries), uint32_t  */
    uintptr_t addr_col_idx; /* CSR column-index array (NNZ entries), uint32_t */
    uintptr_t addr_values;  /* CSR nonzero values (NNZ entries), fp16         */
    uintptr_t addr_x;       /* Input vector x (K entries), fp16               */
    uintptr_t addr_Y;       /* Output vector Y - computed result (M entries)  */
    uintptr_t addr_G;       /* Output vector G - golden model (M entries)     */
    uint32_t M;              /* Rows of A, length of Y / G                     */
    uint32_t K;              /* Columns of A, length of x                      */
    uint32_t nnz;             /* Number of nonzero entries in A                 */
} onnx_spmv_params_t;

#endif /* ONNX_SPMV_PARAMS_H_ */