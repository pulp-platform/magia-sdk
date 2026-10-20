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

void vec_scale_pulp_task(uint32_t data_ptr)
{
    volatile pulp_task_params_t *params = (volatile pulp_task_params_t *)data_ptr;
    volatile int32_t *in_a              = (volatile int32_t *)params->in_a;
    volatile int32_t *out               = (volatile int32_t *)params->out;
    uint32_t len                        = params->len;
    uint32_t scale                      = params->scale;
    uint32_t cluster_local_id           = cl_core_id();

    for (uint32_t i = cluster_local_id; i < len; i++)
        out[i] = in_a[i] * (int32_t)scale;
}
