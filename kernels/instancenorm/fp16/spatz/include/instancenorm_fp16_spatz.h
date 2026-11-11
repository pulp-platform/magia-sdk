#ifndef INSTANCENORM_FP16_SPATZ_H_
#define INSTANCENORM_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_instancenorm_fp16_spatz(const float16 *input,
                                   float16 *output,
                                   const float16 *scale,
                                   const float16 *B,
                                   const float16 epsilon,
                                   uint32_t input_shape[4]);
/* Same operation on this tile only, with input, output, scale and B already in this tile's L1. */
void MAGIA_instancenorm_fp16_spatz_l1(const float16 *input,
                                      float16 *output,
                                      const float16 *scale,
                                      const float16 *B,
                                      const float16 epsilon,
                                      uint32_t input_shape[4]);

#endif /* INSTANCENORM_FP16_SPATZ_H_ */
