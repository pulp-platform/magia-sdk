#ifndef SIGMOID_FP16_SPATZ_H_
#define SIGMOID_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_sigmoid_fp16_spatz(const float16 *X, float16 *Y, uint32_t size);

#endif  /* SIGMOID_FP16_SPATZ_H_ */
