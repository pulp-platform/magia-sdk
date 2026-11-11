// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef EXP_FP16_SPATZ_PARAMS_H_
#define EXP_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_input;  /* Tile's input shard                                       */
    uintptr_t shard_output; /* Tile's computed result shard                             */
    uint32_t start;         /* Tile's shard global start index                          */
    uint32_t end;           /* Tile's shard global end index                            */
    uint32_t len;           /* Tile's shard len                                         */
} exp_fp16_spatz_params_t;

#endif /* EXP_FP16_SPATZ_PARAMS_H_ */
