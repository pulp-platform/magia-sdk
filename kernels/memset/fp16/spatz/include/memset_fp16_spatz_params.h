// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef MEMSET_FP16_SPATZ_PARAMS_H_
#define MEMSET_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_C; /* Tile's output tensor C shard */

    uint32_t len; /* Number of elements */
} memset_fp16_spatz_params_t;

#endif /* MEMSET_FP16_SPATZ_PARAMS_H_ */
