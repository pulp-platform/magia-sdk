#ifndef RMSNORM_FP16_SPATZ_H_
#define RMSNORM_FP16_SPATZ_H_

#include <stdint.h>

/* Y = X / sqrt(mean(X^2) + epsilon) * scale over the last axis of X [rows x w_len] (ONNX
   RMSNormalization, axis = -1): the rows are split over all tiles. */
void MAGIA_rmsnorm_fp16_spatz(const float16 *X,
                              const float16 *scale,
                              const float16 epsilon,
                              uint32_t rows,
                              uint32_t w_len,
                              float16 *Y);
/* Same operation on this tile only, with X, scale and Y already in this tile's L1. Y may alias X. */
void MAGIA_rmsnorm_fp16_spatz_l1(const float16 *X,
                                 const float16 *scale,
                                 const float16 *epsilon,
                                 uint32_t rows,
                                 uint32_t w_len,
                                 float16 *Y);

#endif /* RMSNORM_FP16_SPATZ_H_ */
