// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef REDUCESUM_FP16_SPATZ_PARAMS_H_
#define REDUCESUM_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X; /* Tile's input data shard         */
    uintptr_t shard_Y; /* Tile's output data shard        */

    uint32_t reduce_dim;  /* Elements to collapse (sum over) */
    uint32_t inner_dim;   /* Inner dimension size            */
    uint32_t outer_start; /* Tile's outer slice start        */
    uint32_t outer_len;   /* Tile's outer slice length       */
} reducesum_fp16_spatz_params_t;

#endif /* REDUCESUM_FP16_SPATZ_PARAMS_H_ */
