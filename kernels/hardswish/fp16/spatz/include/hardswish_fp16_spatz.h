#ifndef HARDSWISH_FP16_SPATZ_H_
#define HARDSWISH_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_hardswish_fp16_spatz(const float16 *X, float16 *Y, uint32_t size);
/* Same operation on this tile only, with X, Y already in this tile's L1. Y may alias X. */
void MAGIA_hardswish_fp16_spatz_l1(const float16 *X, float16 *Y, uint32_t size);

#endif /* HARDSWISH_FP16_SPATZ_H_ */
