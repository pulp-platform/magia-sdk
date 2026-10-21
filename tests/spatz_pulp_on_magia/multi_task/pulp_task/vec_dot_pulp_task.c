/*
 * Copyright (C) 2025 ETH Zurich and University of Bologna
 * Licensed under the Apache License, Version 2.0.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Two cores compute a partial dot product over their half of the arrays;
 * each writes its partial result to out[pair_local_id].
 * CV32 checks out[0] + out[1] == expected total dot product.
 */
#include "tile.h"
#include "multi_task_params.h"

void vec_dot_pulp(int32_t *in_a, int32_t *in_b, int32_t *out, uint32_t len)
{
    uint32_t pair_local_id              = cl_core_id() % 2;
    uint32_t half                       = len / 2;
    uint32_t start                      = pair_local_id * half;

    int32_t dot = 0;
    for (uint32_t i = 0; i < half; i++)
        dot += in_a[start + i] * in_b[start + i];

    out[pair_local_id] = dot;

    cl_eu_team_barrier_id(0);
}


void vec_dot_pulp_entry(void *arg){
    pulp_task_params_t *p = (pulp_task_params_t *)arg;
    vec_dot_pulp((int32_t*) p->in_a, (int32_t*) p->in_b, (int32_t*) p->out, (uint32_t) p->len);
}

int vec_dot_pulp_task(void *data) {
    cl_eu_team_barrier_setup(0, (uint32_t) 0x0D);
    cl_eu_team_push_other((uint32_t) 0x0C, vec_dot_pulp_entry, data);
    cl_eu_team_barrier_id(0);
    return 0;
}
