// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef EXPMINUS_FP16_SPATZ_H_
#define EXPMINUS_FP16_SPATZ_H_

#include <stdint.h>

/*
 * Y[r, c] = exp(X[r, c] - M[r]) for a rows x cols matrix X: M holds one value per row, broadcast
 * along the columns. M[r] must be >= every X[r, c] (e.g. the row max).
 */
void MAGIA_expminus_fp16_spatz(
    const float16 *X, const float16 *M, float16 *Y, uint32_t rows, uint32_t cols);
/* Same operation on this tile only, with X, M, Y already in this tile's L1. */
void MAGIA_expminus_fp16_spatz_l1(
    const float16 *X, const float16 *M, float16 *Y, uint32_t rows, uint32_t cols);

#endif /* EXPMINUS_FP16_SPATZ_H_ */
