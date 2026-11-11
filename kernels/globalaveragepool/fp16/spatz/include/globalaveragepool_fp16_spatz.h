#ifndef GLOBALAVERAGEPOOL_FP16_SPATZ_H_
#define GLOBALAVERAGEPOOL_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_globalaveragepool_fp16_spatz(const float16 *X, float16 *Y, uint32_t input_shape[4]);
/* Same operation on this tile only, with X, Y already in this tile's L1. */
void MAGIA_globalaveragepool_fp16_spatz_l1(const float16 *X, float16 *Y, uint32_t input_shape[4]);

#endif /* GLOBALAVERAGEPOOL_FP16_SPATZ_H_ */
