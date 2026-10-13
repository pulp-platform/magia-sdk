#ifndef CONV2D_FP16_SPATZ_H_
#define CONV2D_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_conv2d_fp16_spatz(const float16* X, const float16 *W, const float16 *B, float16 *Y, uint32_t input_shape[4], uint32_t output_shape[4], uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w, uint32_t pad_h, uint32_t pad_w, uint32_t group, int has_bias);

#endif  /* CONV2D_FP16_SPATZ_H_ */
