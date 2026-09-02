#ifndef SOFTMAX_FP16_SPATZ_H_
#define SOFTMAX_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_softmax_fp16_spatz(const float16 *input, float16 *output, uint32_t outer_dim, uint32_t reduce_dim, uint32_t inner_dim);

#endif  /* SOFTMAX_FP16_SPATZ_H_ */
