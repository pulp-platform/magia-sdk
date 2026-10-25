#ifndef BIAS_ROPE_FP16_SPATZ_H_
#define BIAS_ROPE_FP16_SPATZ_H_

#include <stdint.h>

/* X[r, c] += B[c], then (if C is not NULL) the RoPE of each row with the channels of each head
   interleaved, pair i = columns (2i, 2i + 1) by the angle of cos/sin [rows x head_dim / 2] column i.
   The block starts at column col0 (even) of its head. All operands already in this tile's L1. */
void MAGIA_bias_rope_fp16_spatz_l1(float16 *X,
                                   const float16 *B,
                                   const float16 *C,
                                   const float16 *S,
                                   uint32_t rows,
                                   uint32_t cols,
                                   uint32_t col0,
                                   uint32_t head_dim);

#endif /* BIAS_ROPE_FP16_SPATZ_H_ */
