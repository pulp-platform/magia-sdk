#ifndef GELU_FP16_SPATZ_H_
#define GELU_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_gelu_fp16_spatz(const float16 *X, float16 *Y, uint32_t size);

#endif  /* GELU_FP16_SPATZ_H_ */
