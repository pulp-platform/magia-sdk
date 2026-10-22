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

void vec_sum_pulp(int32_t *in_a, int32_t *out, uint32_t len)
{
    // printf("nvm he went in\n");
    volatile uint32_t cid = cl_core_id();
    uint32_t half         = len / 2;
    uint32_t start        = cid * half;
    int32_t sum           = 0;

    for (uint32_t i = 0; i < half; i++)
        sum += in_a[start + i];

    out[cid] = sum;

    return;
}

void vec_sum_pulp_entry(void *arg)
{
    pulp_task_params_t *p = (pulp_task_params_t *)arg;
    vec_sum_pulp((int32_t *)p->in_a, (int32_t *)p->out, (uint32_t)p->len);
}

int vec_sum_pulp_task(void *data)
{
    cl_eu_team_fork(2, vec_sum_pulp_entry, data);
    return 0;
}
