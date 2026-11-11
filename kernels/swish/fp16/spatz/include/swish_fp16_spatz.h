#ifndef SWISH_FP16_SPATZ_H_
#define SWISH_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_swish_fp16_spatz(const float16 *X, float16 *Y, const float16 alpha, uint32_t size);
/* Same operation on this tile only, with X, Y already in this tile's L1. Y may alias X. */
void MAGIA_swish_fp16_spatz_l1(const float16 *X, float16 *Y, const float16 alpha, uint32_t size);

#endif /* SWISH_FP16_SPATZ_H_ */
