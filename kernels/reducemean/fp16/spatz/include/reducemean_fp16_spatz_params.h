// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef REDUCEMEAN_FP16_SPATZ_PARAMS_H_
#define REDUCEMEAN_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X; /* Tile's input data shard         */
    uintptr_t shard_Y; /* Tile's output data shard        */

    uint32_t reduce_dim;  /* Elements to collapse/average    */
    uint32_t inner_dim;   /* Spatial/inner dimension block   */
    uint32_t outer_start; /* Tile's outer loop start index   */
    uint32_t outer_len;   /* Number of outer loops for tile  */
} reducemean_fp16_spatz_params_t;

#endif /* REDUCEMEAN_FP16_SPATZ_PARAMS_H_ */
