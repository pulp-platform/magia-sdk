#ifndef CONCAT_FP16_SPATZ_H_
#define CONCAT_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_concat_fp16_spatz(const float16 *input0, const float16 *input1, float16 *concat_result, uint32_t in0_transfer_len, uint32_t in1_transfer_len, uint32_t axis, uint32_t iterations);

#endif  /* CONCAT_FP16_SPATZ_H_ */
