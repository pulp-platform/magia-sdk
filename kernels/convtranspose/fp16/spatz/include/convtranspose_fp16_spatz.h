// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef CONVTRANSPOSE_FP16_SPATZ_H_
#define CONVTRANSPOSE_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_convtranspose_fp16_spatz(const float16 *X,
                                    const float16 *W,
                                    float16 *Y,
                                    uint32_t input_shape[4],
                                    uint32_t output_shape[4],
                                    uint32_t kernel_h,
                                    uint32_t kernel_w,
                                    uint32_t stride_h,
                                    uint32_t stride_w,
                                    uint32_t pad_h,
                                    uint32_t pad_w,
                                    uint32_t num_groups);
/* Same operation on this tile only, with X, W and Y already in this tile's L1. */
void MAGIA_convtranspose_fp16_spatz_l1(const float16 *X,
                                       const float16 *W,
                                       float16 *Y,
                                       uint32_t input_shape[4],
                                       uint32_t output_shape[4],
                                       uint32_t kernel_h,
                                       uint32_t kernel_w,
                                       uint32_t stride_h,
                                       uint32_t stride_w,
                                       uint32_t pad_h,
                                       uint32_t pad_w,
                                       uint32_t num_groups);

#endif /* CONVTRANSPOSE_FP16_SPATZ_H_ */
