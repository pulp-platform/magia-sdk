// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef TRANSPOSE_FP16_SPATZ_H_
#define TRANSPOSE_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_transpose_fp16_spatz(const float16 *input,
                                float16 *output,
                                uint32_t *perm,
                                uint32_t *in_shape,
                                uint32_t *out_shape,
                                uint32_t rank,
                                uint32_t iterations);
/* Same operation on this tile only, with input, output already in this tile's L1. */
void MAGIA_transpose_fp16_spatz_l1(const float16 *input,
                                   float16 *output,
                                   uint32_t *perm,
                                   uint32_t *in_shape,
                                   uint32_t *out_shape,
                                   uint32_t rank,
                                   uint32_t iterations);

#endif /* TRANSPOSE_FP16_SPATZ_H_ */
