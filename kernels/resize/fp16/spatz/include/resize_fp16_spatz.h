#ifndef RESIZE_FP16_SPATZ_H_
#define RESIZE_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_resize_fp16_spatz(const float16 *X, float16 *Y, uint32_t batch_size, uint32_t channels, uint32_t in_h, uint32_t in_w, uint32_t out_h, uint32_t out_w);

#endif  /* RESIZE_FP16_SPATZ_H_ */
