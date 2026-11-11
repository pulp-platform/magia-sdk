// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef GLOBALMAXPOOL_FP16_SPATZ_PARAMS_H_
#define GLOBALMAXPOOL_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X; /* Tile's input tensor shard    */
    uintptr_t shard_Y; /* Tile's output tensor shard   */
    uint32_t start;    /* Tile's start global index    */
    uint32_t len;      /* Tile's number of elems       */
    uint32_t hw_len;   /* Elements per shard           */
} globalmaxpool_fp16_spatz_params_t;

#endif /* GLOBALMAXPOOL_FP16_SPATZ_PARAMS_H_ */
