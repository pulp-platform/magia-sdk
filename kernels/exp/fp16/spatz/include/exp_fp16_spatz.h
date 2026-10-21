#ifndef EXP_FP16_SPATZ_H_
#define EXP_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_exp_fp16_spatz(const float16 *input, float16 *output, uint32_t size);
/* Same operation on this tile only, with input and output already in this tile's L1. */
void MAGIA_exp_fp16_spatz_l1(const float16 *input, float16 *output, uint32_t n);

#endif /* EXP_FP16_SPATZ_H_ */
