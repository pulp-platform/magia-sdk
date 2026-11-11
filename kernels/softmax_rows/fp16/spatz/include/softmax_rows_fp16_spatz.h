// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SOFTMAX_ROWS_FP16_SPATZ_H_
#define SOFTMAX_ROWS_FP16_SPATZ_H_

#include <stdint.h>

/* X[r, :] = softmax(s * X[r, :]) in place, for a rows x cols matrix X (cols even), with the fast
   exp (exact scaling for s a power of 2). The rows are split over all tiles. */
void MAGIA_softmax_rows_fp16_spatz(float16 *X, float16 s, uint32_t rows, uint32_t cols);
/* Same operation on this tile only, with X already in this tile's L1. */
void MAGIA_softmax_rows_fp16_spatz_l1(float16 *X, float16 s, uint32_t rows, uint32_t cols);

#endif /* SOFTMAX_ROWS_FP16_SPATZ_H_ */
