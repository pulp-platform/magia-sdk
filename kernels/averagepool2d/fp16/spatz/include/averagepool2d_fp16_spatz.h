// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef AVERAGEPOOL2D_FP16_SPATZ_H_
#define AVERAGEPOOL2D_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_averagepool2d_fp16_spatz(const float16 *X,
                                    float16 *Y,
                                    uint32_t input_shape[4],
                                    uint32_t output_shape[4],
                                    uint32_t kernel_h,
                                    uint32_t kernel_w,
                                    uint32_t stride_h,
                                    uint32_t stride_w,
                                    uint32_t pad_h,
                                    uint32_t pad_w);
/* Same operation on this tile only, with X, Y already in this tile's L1. */
void MAGIA_averagepool2d_fp16_spatz_l1(const float16 *X,
                                       float16 *Y,
                                       uint32_t input_shape[4],
                                       uint32_t output_shape[4],
                                       uint32_t kernel_h,
                                       uint32_t kernel_w,
                                       uint32_t stride_h,
                                       uint32_t stride_w,
                                       uint32_t pad_h,
                                       uint32_t pad_w);

#endif /* AVERAGEPOOL2D_FP16_SPATZ_H_ */
