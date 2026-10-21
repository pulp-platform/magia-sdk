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

void vec_absmax_pulp(int32_t *in_a, int32_t *out, uint32_t len)
{
    uint32_t pair_local_id              = cl_core_id() % 2;
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

    cl_eu_team_barrier_id(0);

    return;
}

void vec_absmax_pulp_entry(void *arg){
    pulp_task_params_t *p = (pulp_task_params_t *)arg;
    vec_absmax_pulp((int32_t*) p->in_a, (int32_t*) p->out, (uint32_t) p->len);
}

int vec_absmax_pulp_task(void *data) {
    cl_eu_team_barrier_setup(0, (uint32_t) 0xC1);
    cl_eu_team_push_other((uint32_t) 0xC0, vec_absmax_pulp_entry, data);
    cl_eu_team_barrier_id(0);
    return 0;
}
