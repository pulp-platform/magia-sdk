#ifndef CONV2DGEMM_FP16_SPATZ_H_
#define CONV2DGEMM_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_conv2dgemm_fp16_spatz(const float16 *X,
                                 const float16 *W,
                                 const float16 *B,
                                 float16 *Y,
                                 uint32_t input_shape[4],
                                 uint32_t output_shape[4],
                                 uint32_t kernel_h,
                                 uint32_t kernel_w,
                                 uint32_t stride_h,
                                 uint32_t stride_w,
                                 uint32_t pad_h,
                                 uint32_t pad_w,
                                 uint32_t group,
                                 int has_bias);
/* Same operation on this tile only, with X, W, B and Y already in this tile's L1. scratch is an L1
   buffer of N*C_in*K_h*K_w*H_out*W_out elements for im2col (none for a 1x1, unit-stride, unpadded
   conv) plus C_out*H_out*W_out for the bias (if has_bias). */
void MAGIA_conv2dgemm_fp16_spatz_l1(const float16 *X,
                                    const float16 *W,
                                    const float16 *B,
                                    float16 *Y,
                                    uint32_t input_shape[4],
                                    uint32_t output_shape[4],
                                    uint32_t kernel_h,
                                    uint32_t kernel_w,
                                    uint32_t stride_h,
                                    uint32_t stride_w,
                                    uint32_t pad_h,
                                    uint32_t pad_w,
                                    uint32_t group,
                                    int has_bias,
                                    float16 *scratch);

#endif /* CONV2DGEMM_FP16_SPATZ_H_ */
