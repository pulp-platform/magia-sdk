#ifndef BATCHNORM_FP16_SPATZ_H_
#define BATCHNORM_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_batchnorm_fp16_spatz(const float16 *X,
                                const float16 *scale,
                                const float16 *B,
                                const float16 *input_mean,
                                const float16 *input_var,
                                const float16 epsilon,
                                float16 *Y,
                                uint32_t shape[4]);
/* Same operation on this tile only, with X, scale, B, input_mean, input_var and Y already in this
   tile's L1. */
void MAGIA_batchnorm_fp16_spatz_l1(const float16 *X,
                                   const float16 *scale,
                                   const float16 *B,
                                   const float16 *input_mean,
                                   const float16 *input_var,
                                   const float16 epsilon,
                                   float16 *Y,
                                   uint32_t shape[4]);

#endif /* BATCHNORM_FP16_SPATZ_H_ */
