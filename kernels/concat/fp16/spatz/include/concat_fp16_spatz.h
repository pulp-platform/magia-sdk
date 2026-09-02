#ifndef CONCAT_FP16_SPATZ_H_
#define CONCAT_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_concat_fp16_spatz(const float16 **inputs, const uint32_t *lens, uint32_t num_inputs, float16 *concat_result, uint32_t iterations);

#endif  /* CONCAT_FP16_SPATZ_H_ */
