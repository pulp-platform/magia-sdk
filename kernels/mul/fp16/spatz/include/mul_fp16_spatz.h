#ifndef MUL_FP16_SPATZ_H_
#define MUL_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_mul_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t total_elems);

#endif  /* MUL_FP16_SPATZ_H_ */
