#ifndef HARDSIGMOID_FP16_SPATZ_H_
#define HARDSIGMOID_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_hardsigmoid_fp16_spatz(const float16 *X, float16 *Y, const float16 alpha, const float16 beta, uint32_t size);

#endif  /* HARDSIGMOID_FP16_SPATZ_H_ */
