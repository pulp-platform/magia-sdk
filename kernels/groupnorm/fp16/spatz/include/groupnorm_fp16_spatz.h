#ifndef GROUPNORM_FP16_SPATZ_H_
#define GROUPNORM_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_groupnorm_fp16_spatz(const float16 *X, const float16 *Y, const float16 *scale, const float16 *B, uint32_t num_groups, float16 epsilon);

#endif  /* GROUPNORM_FP16_SPATZ_H_ */
