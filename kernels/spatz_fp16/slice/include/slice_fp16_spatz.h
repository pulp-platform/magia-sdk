#ifndef SLICE_FP16_SPATZ_H_
#define SLICE_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_slice_fp16_spatz(const float16 *data, float16 *sliced, uint32_t outer_dim, uint32_t slice_dim, uint32_t inner_dim, uint32_t start_idx, uint32_t out_slice_dim);

#endif  /* SLICE_FP16_SPATZ_H_ */
