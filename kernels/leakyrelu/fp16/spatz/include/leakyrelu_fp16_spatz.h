#ifndef LEAKYRELU_FP16_SPATZ_H_
#define LEAKYRELU_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_leakyrelu_fp16_spatz(const float16 *X, float16 *Y, uint32_t size, float16 alpha);
/* Same operation on this tile only, with X, Y already in this tile's L1. Y may alias X. */
void MAGIA_leakyrelu_fp16_spatz_l1(const float16 *X, float16 *Y, uint32_t size, float16 alpha);

#endif /* LEAKYRELU_FP16_SPATZ_H_ */
