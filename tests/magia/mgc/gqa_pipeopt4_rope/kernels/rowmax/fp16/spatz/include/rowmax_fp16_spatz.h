#ifndef ROWMAX_FP16_SPATZ_H_
#define ROWMAX_FP16_SPATZ_H_

#include <stdint.h>

/* M[r] = max(max_c X[r, c], P[r]) for a rows x cols matrix X already in this tile's L1;
   P may be NULL, then M[r] = max_c X[r, c]. */
void MAGIA_rowmax_fp16_spatz_l1(
    const float16 *X, const float16 *P, float16 *M, uint32_t rows, uint32_t cols);

#endif /* ROWMAX_FP16_SPATZ_H_ */
