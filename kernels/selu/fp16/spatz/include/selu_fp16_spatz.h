#ifndef SELU_FP16_SPATZ_H_
#define SELU_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_selu_fp16_spatz(
    const float16 *X, float16 *Y, uint32_t size, float16 alpha, float16 gamma);
/* Same operation on this tile only, with X, Y already in this tile's L1. Y may alias X. */
void MAGIA_selu_fp16_spatz_l1(
    const float16 *X, float16 *Y, uint32_t size, float16 alpha, float16 gamma);

#endif /* SELU_FP16_SPATZ_H_ */
