/*
 * Copyright (C) 2025 ETH Zurich and University of Bologna
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Two cores each scale their half of the input by params->scale and write
 * the results to the corresponding half of the output array.
 */
#include "tile.h"
#include "multi_task_params.h"

void vec_scale_pulp(int32_t *in_a, int32_t *out, uint32_t len, uint32_t scale)
{
    uint32_t pair_local_id = cl_core_id() % 2;
    uint32_t half          = len / 2;
    uint32_t start         = pair_local_id * half;

    for (uint32_t i = 0; i < half; i++)
        out[start + i] = in_a[start + i] * (int32_t)scale;

    return;
}

void vec_scale_pulp_entry(void *arg)
{
    pulp_task_params_t *p = (pulp_task_params_t *)arg;
    vec_scale_pulp((int32_t *)p->in_a, (int32_t *)p->out, (uint32_t)p->len, (uint32_t)p->scale);
}

int vec_scale_pulp_task(void *data)
{
    cl_eu_team_barrier_setup(0, (uint32_t)0x31);
    cl_eu_team_push_other((uint32_t)0x31, vec_scale_pulp_entry, data);
    cl_eu_team_barrier_id(0);
    return 0;
}