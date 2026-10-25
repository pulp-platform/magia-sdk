#ifndef SWIGLU_FAST_FP16_SPATZ_H_
#define SWIGLU_FAST_FP16_SPATZ_H_

#include <stdint.h>

/* Y[i] = silu(G[i]) * U[i] for len FP16 elements, in FP16 with the Schraudolph exp (see the task).
   All operands already in this tile's L1; Y may alias G or U. */
void MAGIA_swiglu_fast_fp16_spatz_l1(float16 *Y, const float16 *G, const float16 *U, uint32_t len);

#endif /* SWIGLU_FAST_FP16_SPATZ_H_ */
