#ifndef BIAS_ROPE_FP16_SPATZ_H_
#define BIAS_ROPE_FP16_SPATZ_H_

#include <stdint.h>

/*
 * X[r, c] += B[c], then (if C is not NULL) the RoPE of each row, with the channels of each head in
 * groups of `group`. In group g of a head, columns
 *     (g * group + j, g * group + group / 2 + j)
 * are the pair of angle g * group / 2 + j of C / S [rows x head_dim / 2]. The block starts at
 * column col0 (a multiple of group) of its head. X is updated in place; the rows (tokens) are
 * split over all tiles.
 */
void MAGIA_bias_rope_fp16_spatz(float16 *X,
                                const float16 *B,
                                const float16 *C,
                                const float16 *S,
                                uint32_t rows,
                                uint32_t cols,
                                uint32_t col0,
                                uint32_t head_dim,
                                uint32_t group);
/* Same operation on this tile only, with all operands already in this tile's L1. */
void MAGIA_bias_rope_fp16_spatz_l1(float16 *X,
                                   const float16 *B,
                                   const float16 *C,
                                   const float16 *S,
                                   uint32_t rows,
                                   uint32_t cols,
                                   uint32_t col0,
                                   uint32_t head_dim,
                                   uint32_t group);

#endif /* BIAS_ROPE_FP16_SPATZ_H_ */
