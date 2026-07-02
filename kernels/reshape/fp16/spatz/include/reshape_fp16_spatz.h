#ifndef RESHAPE_FP16_SPATZ_H_
#define RESHAPE_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_reshape_fp16_spatz(const float16 *data, float16 *reshaped, uint32_t total_elements);

#endif  /* RESHAPE_FP16_SPATZ_H_ */
