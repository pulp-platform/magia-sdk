// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ROWDIV_FP16_SPATZ_H_
#define ROWDIV_FP16_SPATZ_H_

#include <stdint.h>

/*
 * Y[r, c] = X[r, c] / S[r] for a rows x cols matrix X, i.e. Y = diag(S)^-1 X: S holds one
 * divisor per row, broadcast along the columns.
 */
void MAGIA_rowdiv_fp16_spatz(
    const float16 *X, const float16 *S, float16 *Y, uint32_t rows, uint32_t cols);
/* Same operation on this tile only, with X, S, Y already in this tile's L1. */
void MAGIA_rowdiv_fp16_spatz_l1(
    const float16 *X, const float16 *S, float16 *Y, uint32_t rows, uint32_t cols);

#endif /* ROWDIV_FP16_SPATZ_H_ */
