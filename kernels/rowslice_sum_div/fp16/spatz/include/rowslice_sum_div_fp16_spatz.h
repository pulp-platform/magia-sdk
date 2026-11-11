// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ROWSLICE_SUM_DIV_FP16_SPATZ_H_
#define ROWSLICE_SUM_DIV_FP16_SPATZ_H_

#include <stdint.h>

/*
 * Y[r, c] = (O[r, c] + sum_{k != own} G[k, r, c]) / S[r], for k < n, with G [n x rows x cols].
 * The rows are split over all tiles.
 */
void MAGIA_rowslice_sum_div_fp16_spatz(float16 *Y,
                                       const float16 *O,
                                       const float16 *G,
                                       const float16 *S,
                                       uint32_t n,
                                       uint32_t own,
                                       uint32_t rows,
                                       uint32_t cols);
/* Same operation on this tile only, with all operands already in this tile's L1. */
void MAGIA_rowslice_sum_div_fp16_spatz_l1(float16 *Y,
                                          const float16 *O,
                                          const float16 *G,
                                          const float16 *S,
                                          uint32_t n,
                                          uint32_t own,
                                          uint32_t rows,
                                          uint32_t cols);

#endif /* ROWSLICE_SUM_DIV_FP16_SPATZ_H_ */
