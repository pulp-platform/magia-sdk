#ifndef IM2COL_FP16_SPATZ_H_
#define IM2COL_FP16_SPATZ_H_

#include <stdint.h>

#include "im2col_fp16_spatz_params.h"

/* im2col of a whole HWC image X [height x width x cin] for a ksize x ksize convolution (ksize odd)
   with stride 1 and (ksize - 1) / 2 zeros on each side: row y * width + x of Y
   [(height * width) x (ksize * ksize * cin)] is the patch of output pixel (y, x), columns in
   (ky, kx, ci) order. The output rows of the image are split over all tiles. */
void MAGIA_im2col_fp16_spatz(
    const float16 *X, float16 *Y, uint32_t height, uint32_t width, uint32_t cin, uint32_t ksize);
/* im2col of one block of output pixels on this tile's Spatz (see the task), all operands already in
   this tile's L1; waits for it. */
void MAGIA_im2col_fp16_spatz_l1(const im2col_fp16_spatz_params_t *params);

#endif /* IM2COL_FP16_SPATZ_H_ */
