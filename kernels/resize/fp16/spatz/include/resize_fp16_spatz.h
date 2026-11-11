// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef RESIZE_FP16_SPATZ_H_
#define RESIZE_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_resize_fp16_spatz(const float16 *X,
                             float16 *Y,
                             uint32_t batch_size,
                             uint32_t channels,
                             uint32_t in_h,
                             uint32_t in_w,
                             uint32_t out_h,
                             uint32_t out_w);
/* Same operation on this tile only, with X, Y already in this tile's L1. */
void MAGIA_resize_fp16_spatz_l1(const float16 *X,
                                float16 *Y,
                                uint32_t batch_size,
                                uint32_t channels,
                                uint32_t in_h,
                                uint32_t in_w,
                                uint32_t out_h,
                                uint32_t out_w);

#endif /* RESIZE_FP16_SPATZ_H_ */
