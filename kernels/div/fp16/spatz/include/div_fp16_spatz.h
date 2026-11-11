// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef DIV_FP16_SPATZ_H_
#define DIV_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_div_fp16_spatz(const float16 *A, const float16 *B, float16 *C, uint32_t size);
/* Same operation on this tile only, with A, B, C already in this tile's L1. C may alias A or B. */
void MAGIA_div_fp16_spatz_l1(const float16 *A, const float16 *B, float16 *C, uint32_t size);

#endif /* DIV_FP16_SPATZ_H_ */
