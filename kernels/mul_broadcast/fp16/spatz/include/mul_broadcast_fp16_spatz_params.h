// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef MUL_BROADCAST_FP16_SPATZ_PARAMS_H_
#define MUL_BROADCAST_FP16_SPATZ_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t shard_A; /* Tile's input tensor A shard        */
    uintptr_t shard_C; /* Tile's output tensor C shard       */
    uintptr_t scalar;  /* L1 address of the scalar multiplier */

    uint32_t total_elems; /* Total elements in the tensor */
    uint32_t elem_start;  /* Tile's element shard start   */
    uint32_t elem_len;    /* Tile's number of elements    */
} mul_broadcast_fp16_spatz_params_t;

#endif /* MUL_BROADCAST_FP16_SPATZ_PARAMS_H_ */
