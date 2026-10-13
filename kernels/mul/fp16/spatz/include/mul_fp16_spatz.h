#ifndef MUL_FP16_SPATZ_H_
#define MUL_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_mul_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t total_elems);

/* Broadcast form, A = [rows, row_len]. mode is MUL_BCAST_ROW (B is one row_len row,
 * shared by every row) or MUL_BCAST_SCALAR (B is one scalar per row) - see
 * mul_bcast_fp16_spatz_params.h. */
void MAGIA_mul_bcast_fp16_spatz(const float16 *A, const float16 *B, float16 *Y, uint32_t rows, uint32_t row_len, uint32_t mode);

#endif  /* MUL_FP16_SPATZ_H_ */
