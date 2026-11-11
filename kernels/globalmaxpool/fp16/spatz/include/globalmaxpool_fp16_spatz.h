// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef GLOBALMAXPOOL_FP16_SPATZ_H_
#define GLOBALMAXPOOL_FP16_SPATZ_H_

#include <stdint.h>

void MAGIA_globalmaxpool_fp16_spatz(const float16 *X, float16 *Y, uint32_t input_shape[4]);
/* Same operation on this tile only, with X, Y already in this tile's L1. */
void MAGIA_globalmaxpool_fp16_spatz_l1(const float16 *X, float16 *Y, uint32_t input_shape[4]);

#endif /* GLOBALMAXPOOL_FP16_SPATZ_H_ */
