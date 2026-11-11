// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef HARDSIGMOID_FP16_SPATZ_H_
#define HARDSIGMOID_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_hardsigmoid_fp16_spatz(
    const float16 *X, float16 *Y, const float16 alpha, const float16 beta, uint32_t size);
/* Same operation on this tile only, with X, Y already in this tile's L1. Y may alias X. */
void MAGIA_hardsigmoid_fp16_spatz_l1(
    const float16 *X, float16 *Y, const float16 alpha, const float16 beta, uint32_t size);

#endif /* HARDSIGMOID_FP16_SPATZ_H_ */
