#ifndef RELU_FP16_SPATZ_H_
#define RELU_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_relu_fp16_spatz(const float16 *X, float16 *Y, uint32_t size);

#endif  /* RELU_FP16_SPATZ_H_ */
