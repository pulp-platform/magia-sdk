#ifndef GATHER_FP16_SPATZ_H_
#define GATHER_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_gather_fp16_spatz(const float16 *data, uint32_t in_shape[4], float16 *output, uint32_t batch, uint32_t gather_dim_size, uint32_t axis_length, uint32_t index, uint32_t axis);

#endif  /* GATHER_FP16_SPATZ_H_ */
