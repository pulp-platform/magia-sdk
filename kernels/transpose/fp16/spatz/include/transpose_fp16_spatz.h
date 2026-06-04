#ifndef TRANSPOSE_FP16_SPATZ_H_
#define TRANSPOSE_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_transpose_fp16_spatz(const float16 *input, float16 *output, uint32_t *perm, uint32_t iterations);

#endif  /* TRANSPOSE_FP16_SPATZ_H_ */
