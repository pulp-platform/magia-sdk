// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef COLSUM_RESCALE_FP16_SPATZ_H_
#define COLSUM_RESCALE_FP16_SPATZ_H_

#include <stdint.h>

/*
 * S[c] = S[c] + sum_{r >= 1} G[r, c]; if P is not NULL, then P[c] = A[c] * P[c] and
 * S[c] = P[c] + S[c] (cols even). The columns are split over all tiles, in pairs.
 */
void MAGIA_colsum_rescale_fp16_spatz(
    float16 *S, const float16 *G, float16 *P, const float16 *A, uint32_t rows, uint32_t cols);
/* Same operation on this tile only, with all operands already in this tile's L1. */
void MAGIA_colsum_rescale_fp16_spatz_l1(
    float16 *S, const float16 *G, float16 *P, const float16 *A, uint32_t rows, uint32_t cols);

#endif /* COLSUM_RESCALE_FP16_SPATZ_H_ */
