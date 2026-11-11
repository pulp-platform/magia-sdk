// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef LAYERNORM_FP16_SPATZ_H_
#define LAYERNORM_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_layernorm_fp16_spatz(const float16 *X,
                                const float16 *scale,
                                const float16 *B,
                                const float16 epsilon,
                                uint32_t *input_shape,
                                uint32_t rank,
                                float16 *Y);
/* Same operation on this tile only, with X, scale, B and Y already in this tile's L1. */
void MAGIA_layernorm_fp16_spatz_l1(const float16 *X,
                                   const float16 *scale,
                                   const float16 *B,
                                   const float16 epsilon,
                                   uint32_t *input_shape,
                                   uint32_t rank,
                                   float16 *Y);

#endif /* LAYERNORM_FP16_SPATZ_H_ */
