#ifndef ROPE_FP16_SPATZ_H_
#define ROPE_FP16_SPATZ_H_

#include <stdint.h>

/* Rotary position embedding, rotate-half form (Llama, ONNX RotaryEmbedding with interleaved = 0):
   X [rows x n_heads * head_dim], one token per row; cos, sin [rows x head_dim / 2], already
   gathered at each row's position. For every head x = [x1 | x2]:
     Y = [x1 * cos - x2 * sin | x2 * cos + x1 * sin]
   The rows are split over all tiles. */
void MAGIA_rope_fp16_spatz(const float16 *X,
                           const float16 *cos,
                           const float16 *sin,
                           uint32_t rows,
                           uint32_t n_heads,
                           uint32_t head_dim,
                           float16 *Y);
/* Same operation on this tile only, with X, cos, sin and Y already in this tile's L1. Y may
   alias X. */
void MAGIA_rope_fp16_spatz_l1(const float16 *X,
                              const float16 *cos,
                              const float16 *sin,
                              uint32_t rows,
                              uint32_t n_heads,
                              uint32_t head_dim,
                              float16 *Y);

#endif /* ROPE_FP16_SPATZ_H_ */
