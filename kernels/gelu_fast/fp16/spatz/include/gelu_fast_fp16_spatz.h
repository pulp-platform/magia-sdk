#ifndef GELU_FAST_FP16_SPATZ_H_
#define GELU_FAST_FP16_SPATZ_H_

#include <stdint.h>

/* Y[i] = gelu(X[i]) for len FP16 elements, tanh form, in FP16 with the Schraudolph exp (see the
   task). The elements are split over all tiles. */
void MAGIA_gelu_fast_fp16_spatz(float16 *Y, const float16 *X, uint32_t len);
/* Same operation on this tile only, with X and Y already in this tile's L1; Y may alias X. */
void MAGIA_gelu_fast_fp16_spatz_l1(float16 *Y, const float16 *X, uint32_t len);

#endif /* GELU_FAST_FP16_SPATZ_H_ */
