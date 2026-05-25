#ifndef INSTANCENORM_FP16_SPATZ_H_
#define INSTANCENORM_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_instancenorm_fp16_spatz(const float16 *input, float16 *output, const float16 *scale, const float16 *B, const float16 epsilon);

#endif  /* INSTANCENORM_FP16_SPATZ_H_ */
