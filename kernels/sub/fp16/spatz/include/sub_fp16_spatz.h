#ifndef SUB_FP16_SPATZ_H_
#define SUB_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_sub_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t size);
/* Same operation on this tile only, with A, B, C already in this tile's L1. */
void MAGIA_sub_fp16_spatz_l1(const float16 *A, const float16 *B, float16 *C, uint32_t n);

#endif /* SUB_FP16_SPATZ_H_ */
