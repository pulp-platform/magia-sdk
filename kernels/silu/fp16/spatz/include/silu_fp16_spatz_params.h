// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SILU_FP16_SPATZ_PARAMS_H_
#define SILU_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_X; /* Tile's input tensor shard                   */
    uintptr_t shard_Y; /* Tile's output tensor shard (may alias X)    */

    uint32_t total_elems; /* Total elements in the tensor */
    uint32_t elem_start;  /* Tile's element shard start   */
    uint32_t elem_len;    /* Tile's number of elements    */
} silu_fp16_spatz_params_t;

#endif /* SILU_FP16_SPATZ_PARAMS_H_ */
