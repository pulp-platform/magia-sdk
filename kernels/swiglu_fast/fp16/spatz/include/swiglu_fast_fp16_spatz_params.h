// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SWIGLU_FAST_FP16_SPATZ_PARAMS_H_
#define SWIGLU_FAST_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_G; /* Gate [len], overwritten with the output if shard_Y == shard_G */
    uintptr_t shard_U; /* Up [len]                                                    */
    uintptr_t shard_Y; /* Output [len] (may alias G, U)                               */

    uint32_t len;      /* Number of elements */
} swiglu_fast_fp16_spatz_params_t;

#endif /* SWIGLU_FAST_FP16_SPATZ_PARAMS_H_ */
