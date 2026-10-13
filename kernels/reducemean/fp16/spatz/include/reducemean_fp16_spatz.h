#ifndef REDUCEMEAN_FP16_SPATZ_H_
#define REDUCEMEAN_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_reducemean_fp16_spatz(const float16 *X, float16 *Y, uint32_t outer_dim, uint32_t reduce_dim, uint32_t inner_dim);

#endif  /* REDUCEMEAN_FP16_SPATZ_H_ */
