/*
 * Copyright (C) 2026 ETH Zurich and University of Bologna
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef FORK_TASK_PARAMS_H_
#define FORK_TASK_PARAMS_H_

#include <stdint.h>

typedef struct {
    uint32_t in_a;
    uint32_t in_b;
    uint32_t out;
    uint32_t len;
} pulp_task_params_t;

#endif /* FORK_TASK_PARAMS_H_ */