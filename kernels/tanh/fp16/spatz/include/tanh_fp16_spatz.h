// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef TANH_FP16_SPATZ_H_
#define TANH_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_tanh_fp16_spatz(const float16 *input, float16 *output, uint32_t size);
/*
 * Same operation on this tile only, with input, output already in this tile's L1. output may alias
 * input.
 */
void MAGIA_tanh_fp16_spatz_l1(const float16 *input, float16 *output, uint32_t size);

#endif /* TANH_FP16_SPATZ_H_ */
