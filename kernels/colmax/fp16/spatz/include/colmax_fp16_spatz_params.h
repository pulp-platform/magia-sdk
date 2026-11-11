// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef COLMAX_FP16_SPATZ_PARAMS_H_
#define COLMAX_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X; /* Input matrix [rows x cols] */
    uintptr_t shard_M; /* Output column max [cols]   */

    uint32_t dims; /* rows | (row length << 16) */
} colmax_fp16_spatz_params_t;

#endif /* COLMAX_FP16_SPATZ_PARAMS_H_ */
