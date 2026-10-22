// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SOFTMAX_ROWS_FP16_SPATZ_PARAMS_H_
#define SOFTMAX_ROWS_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X; /* Matrix [rows x cols], softmax of each row of s * X in place */
    uint32_t dims;     /* Number of rows | row length << 16                          */
    uint32_t min_coef; /* fp16 bits of the exp clamp MIN / s | COEF * s << 16        */
} softmax_rows_fp16_spatz_params_t;

/* Fast exp (Schraudolph 1999): exp(x) ~ reinterpret_fp16(COEF * x + BIAS), with x clamped to
   MIN from below, as in flat_attn_pipeopt's expminus_rowsum. */
#define SOFTMAX_ROWS_EXP_MIN  (-10.0f)
#define SOFTMAX_ROWS_EXP_COEF (1477.0f)
#define SOFTMAX_ROWS_EXP_BIAS (15360.0f)

#endif /* SOFTMAX_ROWS_FP16_SPATZ_PARAMS_H_ */
