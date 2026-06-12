#ifndef SCATTER_FP16_SPATZ_H_
#define SCATTER_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_scatter_fp16_spatz(const float16 *data, const int64_t *indices, const float16 *updates, float16 *output, uint32_t outer_size, uint32_t inner_size, uint32_t axis, uint32_t data_axis_dim, uint32_t indices_axis_dim);

#endif  /* SCATTER_FP16_SPATZ_H_ */
