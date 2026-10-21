#ifndef SWIGLU_FP16_SPATZ_H_
#define SWIGLU_FP16_SPATZ_H_

#include <stdint.h>

/* Y = SiLU(G) * U = G / (1 + exp(-G)) * U (the gating of a SwiGLU FFN, G = X @ W_gate,
   U = X @ W_up), elementwise over total_elems elements, split over all tiles. */
void MAGIA_swiglu_fp16_spatz(const float16 *G, const float16 *U, float16 *Y, uint32_t total_elems);
/* Same operation on this tile only, with G, U and Y already in this tile's L1. Y may alias G or U. */
void MAGIA_swiglu_fp16_spatz_l1(const float16 *G, const float16 *U, float16 *Y, uint32_t n);

#endif /* SWIGLU_FP16_SPATZ_H_ */
