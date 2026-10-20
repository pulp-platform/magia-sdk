/*
 * Copyright (C) 2025 ETH Zurich and University of Bologna
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Two cores split the array in half; each writes its partial sum to out[pair_local_id].
 * CV32 checks out[0] + out[1] == expected total sum.
 */
#include "tile.h"
#include "multi_task_params.h"

void vec_sum_pulp_task(uint32_t data_ptr)
{
    volatile pulp_task_params_t *params = (volatile pulp_task_params_t *)data_ptr;
    volatile int32_t *in_a              = (volatile int32_t *)params->in_a;
    volatile int32_t *out               = (volatile int32_t *)params->out;
    uint32_t len                        = params->len;
    uint32_t core_local_id              = cl_core_id();

    int32_t sum = 0;
    for (uint32_t i = core_local_id; i < len; i++)
        sum += in_a[i];

    out[core_local_id] = sum;
}
