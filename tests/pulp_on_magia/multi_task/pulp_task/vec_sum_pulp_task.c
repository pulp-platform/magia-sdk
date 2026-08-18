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
    uint32_t pair_local_id              = GET_PULP_LOCAL_ID(get_hartid()) % 2;
    uint32_t half                       = len / 2;
    uint32_t start                      = pair_local_id * half;

    int32_t sum = 0;
    for (uint32_t i = 0; i < half; i++)
        sum += in_a[start + i];

    out[pair_local_id] = sum;
}
