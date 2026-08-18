/*
 * Copyright (C) 2025 ETH Zurich and University of Bologna
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Two cores each find the absolute maximum of their half of the array and
 * write the result to out[pair_local_id].
 * CV32 checks max(out[0], out[1]) == expected global absmax.
 */
#include "tile.h"
#include "multi_task_params.h"

void vec_absmax_pulp_task(uint32_t data_ptr)
{
    volatile pulp_task_params_t *params = (volatile pulp_task_params_t *)data_ptr;
    volatile int32_t *in_a              = (volatile int32_t *)params->in_a;
    volatile int32_t *out               = (volatile int32_t *)params->out;
    uint32_t len                        = params->len;
    uint32_t pair_local_id              = GET_PULP_LOCAL_ID(get_hartid()) % 2;
    uint32_t half                       = len / 2;
    uint32_t start                      = pair_local_id * half;

    int32_t absmax = 0;
    for (uint32_t i = 0; i < half; i++) {
        int32_t v  = in_a[start + i];
        int32_t av = v < 0 ? -v : v;
        if (av > absmax)
            absmax = av;
    }

    out[pair_local_id] = absmax;
}
