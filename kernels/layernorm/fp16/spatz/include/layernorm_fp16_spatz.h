#ifndef LAYERNORM_FP16_SPATZ_H_
#define LAYERNORM_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_layernorm_fp16_spatz(const float16 *X, const float16 *scale, const float16 *B, const float16 epsilon, uint32_t input_shape[4], float16 *Y);

#endif  /* LAYERNORM_FP16_SPATZ_H_ */
