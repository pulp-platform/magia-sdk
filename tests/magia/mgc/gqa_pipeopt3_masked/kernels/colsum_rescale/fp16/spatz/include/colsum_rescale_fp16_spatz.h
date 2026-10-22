#ifndef COLSUM_RESCALE_FP16_SPATZ_H_
#define COLSUM_RESCALE_FP16_SPATZ_H_

#include <stdint.h>

/* S[c] = S[c] + sum_{r >= 1} G[r, c]; if P is not NULL, then P[c] = A[c] * P[c] and
   S[c] = P[c] + S[c]. All operands already in this tile's L1. */
void MAGIA_colsum_rescale_fp16_spatz_l1(
    float16 *S, const float16 *G, float16 *P, const float16 *A, uint32_t rows, uint32_t cols);

#endif /* COLSUM_RESCALE_FP16_SPATZ_H_ */
