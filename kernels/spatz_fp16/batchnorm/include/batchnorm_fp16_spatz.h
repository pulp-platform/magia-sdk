#ifndef BATCHNORM_FP16_SPATZ_H_
#define BATCHNORM_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_batchnorm_fp16_spatz(const float16 *X, const float16 *scale, const float16 *B, const float16 *input_mean, const float16* input_var, const float16 epsilon, float16 *Y, uint32_t shape[4]);

#endif  /* BATCHNORM_FP16_SPATZ_H_ */
