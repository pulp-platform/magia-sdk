/*
 * Copyright (C) 2025 ETH Zurich and University of Bologna
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef MULTI_TASK_PARAMS_H_
#define MULTI_TASK_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t A; /* fp16 input array pointer  */
    uintptr_t B; /* fp16 input array pointer  */
    uintptr_t C; /* fp16 output array pointer */
    uint32_t len;
} spatz_add_params_t;

typedef struct {
    uintptr_t X; /* fp16 input array pointer  */
    uintptr_t Y; /* fp16 output array pointer */
    uint32_t len;
} spatz_relu_params_t;

typedef struct {
    uintptr_t in_a; /* int32_t input array A */
    uintptr_t in_b; /* int32_t input array B (dot product only) */
    uintptr_t out;  /* int32_t output array  */
    uint32_t len;
    uint32_t scale; /* scale factor (vec_scale only) */
} pulp_task_params_t;

#endif /* MULTI_TASK_PARAMS_H_ */
