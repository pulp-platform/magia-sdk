#ifndef ADD_FP16_SPATZ_H_
#define ADD_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_add_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t size);
/* Same operation on this tile only, with A, B, C already in this tile's L1. */
void MAGIA_add_fp16_spatz_l1(const float16 *A, const float16 *B, float16 *C, uint32_t n);

#endif /* ADD_FP16_SPATZ_H_ */
