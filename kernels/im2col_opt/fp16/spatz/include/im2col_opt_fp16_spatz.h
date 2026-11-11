// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef IM2COL_OPT_FP16_SPATZ_H_
#define IM2COL_OPT_FP16_SPATZ_H_

#include <stdint.h>

#include "im2col_opt_fp16_spatz_params.h"

/*
 * im2col of a whole HWC image X [height x width x cin] for a ksize x ksize convolution (ksize odd)
 * with stride `stride` and pad = (ksize - 1) / 2 zeros on each side. With
 *     out_h = (height + 2 * pad - ksize) / stride + 1, and out_w likewise,
 * row y * out_w + x of Y [(out_h * out_w) x (ksize * ksize * cin)] is the patch of output pixel
 * (y, x), columns in (ky, kx, ci) order. The output rows are split over all tiles.
 */
void MAGIA_im2col_opt_fp16_spatz(const float16 *X,
                                 float16 *Y,
                                 uint32_t height,
                                 uint32_t width,
                                 uint32_t cin,
                                 uint32_t ksize,
                                 uint32_t stride);
/*
 * im2col of one block of output pixels on this tile's Spatz (see the task), with all operands
 * already in this tile's L1. Waits for the task.
 */
void MAGIA_im2col_opt_fp16_spatz_l1(const im2col_opt_fp16_spatz_params_t *params);

#endif /* IM2COL_OPT_FP16_SPATZ_H_ */
