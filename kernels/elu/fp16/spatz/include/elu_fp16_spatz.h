#ifndef ELU_FP16_SPATZ_H_
#define ELU_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_elu_fp16_spatz(const float16 *X, float16 *Y, uint32_t size, float16 alpha);

#endif  /* ELU_FP16_SPATZ_H_ */
