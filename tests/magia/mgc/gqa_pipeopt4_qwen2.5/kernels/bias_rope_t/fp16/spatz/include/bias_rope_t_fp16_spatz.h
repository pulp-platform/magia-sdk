#ifndef BIAS_ROPE_T_FP16_SPATZ_H_
#define BIAS_ROPE_T_FP16_SPATZ_H_

#include <stdint.h>

/* X[r, t] += B[r], then the RoPE of each column (token) with the channels of each head
   interleaved, pair i = rows (2i, 2i + 1) by the angle of cos/sin [head_dim / 2 x cols] row i.
   The block starts at row row0 (even) of its head. All operands already in this tile's L1. */
void MAGIA_bias_rope_t_fp16_spatz_l1(float16 *X,
                                     const float16 *B,
                                     const float16 *C,
                                     const float16 *S,
                                     uint32_t rows,
                                     uint32_t cols,
                                     uint32_t row0,
                                     uint32_t head_dim);

#endif /* BIAS_ROPE_T_FP16_SPATZ_H_ */
