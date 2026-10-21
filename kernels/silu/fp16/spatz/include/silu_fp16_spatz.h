#ifndef SILU_FP16_SPATZ_H_
#define SILU_FP16_SPATZ_H_

#include <stdint.h>

/* Y = X * sigmoid(X) = X / (1 + exp(-X)) (ONNX Swish with alpha = 1), elementwise over
   total_elems elements, split over all tiles. */
void MAGIA_silu_fp16_spatz(const float16 *X, float16 *Y, uint32_t total_elems);
/* Same operation on this tile only, with X and Y already in this tile's L1. Y may alias X. */
void MAGIA_silu_fp16_spatz_l1(const float16 *X, float16 *Y, uint32_t n);

#endif /* SILU_FP16_SPATZ_H_ */
