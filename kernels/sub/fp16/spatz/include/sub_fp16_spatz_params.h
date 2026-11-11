// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SUB_FP16_PARAMS_H_
#define SUB_FP16_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_A; /* Tile's first operand shard       */
    uintptr_t shard_B; /* Tile's second operand shard      */
    uintptr_t shard_C; /* Tile's computed result shard     */
    uint32_t start;    /* Tile's shard global start index  */
    uint32_t end;      /* Tile's shard global end index    */
    uint32_t len;      /* Tile's shard len                 */
} sub_fp16_spatz_params_t;

#endif /* SUB_FP16_PARAMS_H_ */
