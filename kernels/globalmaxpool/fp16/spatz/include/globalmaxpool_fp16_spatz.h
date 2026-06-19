#ifndef GLOBALMAXPOOL_FP16_SPATZ_H_
#define GLOBALMAXPOOL_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_globalmaxpool_fp16_spatz(const float16 *X, float16 *Y, uint32_t input_shape[4]);

#endif  /* GLOBALMAXPOOL_FP16_SPATZ_H_ */
