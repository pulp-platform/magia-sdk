// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdbool.h>
#include <stdint.h>

#include "tile.h"
#include "eventunit.h"

#include "group_normalize_fp16_spatz_params.h"
#include "group_normalize_fp16_spatz_task_bin.h"
#include "kernel_test_utils.h"

#define CHANNELS 128u
#define SPATIAL_ELEMENTS 64u
#define ELEMENTS (CHANNELS * SPATIAL_ELEMENTS)
#define MAX_TASK_CYCLES 300000u

extern uint32_t _spatz_binary_start;

static inline uint32_t read_cycles(void)
{
    uint32_t cycles;
    asm volatile("rdcycle %0" : "=r"(cycles));
    return cycles;
}

static int wait_for_spatz(void)
{
    eu_config_t config = {.hartid = get_hartid()};
    eu_controller_t controller = {
        .base = 0,
        .cfg = &config,
        .api = &eu_api,
    };
    if (eu_spatz_wait(&controller, WFE) == 0)
        return -1;
    return (int)spatz_get_exit_code();
}

int main(void)
{
    l1_alloc_init();
    group_normalize_fp16_spatz_params_t *params =
        l1_alloc(sizeof(group_normalize_fp16_spatz_params_t));
    float16 *input = l1_alloc(ELEMENTS * sizeof(float16));
    float16 *output = l1_alloc(ELEMENTS * sizeof(float16));
    float16 *mean = l1_alloc(sizeof(float16));
    float16 *variance = l1_alloc(sizeof(float16));
    float16 *scale = l1_alloc(CHANNELS * sizeof(float16));
    float16 *bias = l1_alloc(CHANNELS * sizeof(float16));

    *mean = (float16)0.125f;
    *variance = (float16)1.5f;
    for (uint32_t channel = 0; channel < CHANNELS; ++channel) {
        scale[channel] = (float16)(0.5f + (float)(channel % 7u) * 0.125f);
        bias[channel] = (float16)((float)((int32_t)(channel % 5u) - 2) * 0.0625f);
    }
    for (uint32_t index = 0; index < ELEMENTS; ++index)
        input[index] = (float16)((float)((int32_t)(index % 23u) - 11) * 0.0625f);

    params->input = (uintptr_t)input;
    params->mean = (uintptr_t)mean;
    params->variance = (uintptr_t)variance;
    params->scale = (uintptr_t)scale;
    params->bias = (uintptr_t)bias;
    params->output = (uintptr_t)output;
    params->local_spatial_elements = SPATIAL_ELEMENTS;
    params->local_elements = ELEMENTS;
    params->channel_offset = 0u;
    params->scale_channel_offset = 0u;
    params->num_groups = 1u;
    params->channels_per_group = CHANNELS;
    params->epsilon = 1.00135803e-05f;

    kt_spatz_init((uint32_t)&_spatz_binary_start);
    uint32_t start = read_cycles();
    spatz_run_task_with_params(
        GROUP_NORMALIZE_FP16_SPATZ_TASK, (uint32_t)params);
    int result = wait_for_spatz();
    uint32_t cycles = read_cycles() - start;

    float stddev;
    asm volatile("fsqrt.s %0, %1"
                 : "=f"(stddev)
                 : "f"((float)*variance + params->epsilon));
    const float inverse_stddev = 1.0f / stddev;
    bool correct = result == 0;
    for (uint32_t index = 0; index < ELEMENTS; ++index) {
        const uint32_t channel = index / SPATIAL_ELEMENTS;
        const float16 expected = (float16)(
            ((float)input[index] - (float)*mean) * inverse_stddev *
                (float)scale[channel] + (float)bias[channel]);
        correct &= kt_bits(output[index]) == kt_bits(expected);
    }
    const bool fast_enough = cycles < MAX_TASK_CYCLES;
    printf("[CV32 (%d)] group-normalize cycles=%u limit=%u output=%s performance=%s\n",
           get_hartid(), cycles, MAX_TASK_CYCLES,
           correct ? "SUCCESS" : "FAILED",
           fast_enough ? "SUCCESS" : "FAILED");
    spatz_clk_dis();
    return correct && fast_enough ? 0 : -1;
}
