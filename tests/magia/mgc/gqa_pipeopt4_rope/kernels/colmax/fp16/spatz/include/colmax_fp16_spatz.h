#ifndef COLMAX_FP16_SPATZ_H_
#define COLMAX_FP16_SPATZ_H_

#include <stdint.h>

/* M[c] = max_r X[r, c] for a rows x cols matrix X already in this tile's L1. */
void MAGIA_colmax_fp16_spatz_l1(const float16 *X, float16 *M, uint32_t rows, uint32_t cols);

#endif /* COLMAX_FP16_SPATZ_H_ */
