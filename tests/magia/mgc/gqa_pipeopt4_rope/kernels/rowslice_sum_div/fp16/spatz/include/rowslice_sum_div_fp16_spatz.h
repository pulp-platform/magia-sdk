#ifndef ROWSLICE_SUM_DIV_FP16_SPATZ_H_
#define ROWSLICE_SUM_DIV_FP16_SPATZ_H_

#include <stdint.h>

/* Y[r, c] = (O[r, c] + sum_{k != own} G[k, r, c]) / S[r], for k < n. All operands already in
   this tile's L1. */
void MAGIA_rowslice_sum_div_fp16_spatz_l1(float16 *Y,
                                          const float16 *O,
                                          const float16 *G,
                                          const float16 *S,
                                          uint32_t n,
                                          uint32_t own,
                                          uint32_t rows,
                                          uint32_t cols);

#endif /* ROWSLICE_SUM_DIV_FP16_SPATZ_H_ */
