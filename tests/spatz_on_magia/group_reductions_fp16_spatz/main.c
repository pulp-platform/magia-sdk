// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdbool.h>
#include <stdint.h>

#include "tile.h"
#include "eventunit.h"

#include "group_centered_reduce_fp16_spatz_params.h"
#include "group_reduce_fp16_spatz_params.h"
#include "group_reductions_fp16_spatz_task_bin.h"
#include "kernel_test_utils.h"

#define CHANNELS 128u
#define SPATIAL_ELEMENTS 64u
#define ELEMENTS (CHANNELS * SPATIAL_ELEMENTS)
#define MAX_REDUCTION_CYCLES 20000u

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

static float16 reference_mean(const float16 *input, uint32_t elements,
                              uint32_t elements_per_group)
{
    float16 lanes[256];
    for (uint32_t lane = 0; lane < 256u; ++lane)
        lanes[lane] = 0.0f;
    const float16 scale = (float16)(1.0f / (float)elements_per_group);
    for (uint32_t index = 0; index < elements; ++index) {
        const uint32_t lane = index & 255u;
        lanes[lane] = (float16)(lanes[lane] + (float16)(input[index] * scale));
    }
    float16 sum = 0.0f;
    for (uint32_t lane = 0; lane < 256u; ++lane)
        sum = (float16)(sum + lanes[lane]);
    return sum;
}

static float16 reference_variance(const float16 *input, uint32_t elements,
                                  uint32_t elements_per_group, float16 mean)
{
    float16 lanes[256];
    for (uint32_t lane = 0; lane < 256u; ++lane)
        lanes[lane] = 0.0f;
    const float16 scale = (float16)(1.0f / (float)elements_per_group);
    for (uint32_t index = 0; index < elements; ++index) {
        const float16 centered = (float16)(input[index] - mean);
        const uint32_t lane = index & 255u;
        lanes[lane] = (float16)(lanes[lane] +
            (float16)((float16)(centered * scale) * centered));
    }
    float16 sum = 0.0f;
    for (uint32_t lane = 0; lane < 256u; ++lane)
        sum = (float16)(sum + lanes[lane]);
    return sum;
}

int main(void)
{
    l1_alloc_init();
    float16 *input = l1_alloc(ELEMENTS * sizeof(float16));
    float16 *mean = l1_alloc(2u * sizeof(float16));
    float16 *variance = l1_alloc(2u * sizeof(float16));
    group_reduce_fp16_spatz_params_t *mean_params =
        l1_alloc(sizeof(group_reduce_fp16_spatz_params_t));
    group_centered_reduce_fp16_spatz_params_t *variance_params =
        l1_alloc(sizeof(group_centered_reduce_fp16_spatz_params_t));

    for (uint32_t index = 0; index < ELEMENTS; ++index)
        input[index] = (float16)((float)((int32_t)(index % 29u) - 14) * 0.0625f);

    mean_params->input = (uintptr_t)input;
    mean_params->output = (uintptr_t)mean;
    mean_params->local_elements = ELEMENTS;
    mean_params->local_spatial_elements = SPATIAL_ELEMENTS;
    mean_params->channel_offset = 0u;
    mean_params->num_groups = 1u;
    mean_params->elements_per_group = ELEMENTS;
    mean_params->channels_per_group = CHANNELS;

    variance_params->input = (uintptr_t)input;
    variance_params->mean = (uintptr_t)mean;
    variance_params->output = (uintptr_t)variance;
    variance_params->local_elements = ELEMENTS;
    variance_params->local_spatial_elements = SPATIAL_ELEMENTS;
    variance_params->channel_offset = 0u;
    variance_params->num_groups = 1u;
    variance_params->elements_per_group = ELEMENTS;
    variance_params->channels_per_group = CHANNELS;

    kt_spatz_init((uint32_t)&_spatz_binary_start);
    uint32_t start = read_cycles();
    spatz_run_task_with_params(
        GROUP_REDUCE_FP16_SPATZ_TASK, (uint32_t)mean_params);
    int mean_result = wait_for_spatz();
    uint32_t mean_cycles = read_cycles() - start;

    const float16 expected_mean = reference_mean(input, ELEMENTS, ELEMENTS);
    start = read_cycles();
    spatz_run_task_with_params(
        GROUP_CENTERED_REDUCE_FP16_SPATZ_TASK, (uint32_t)variance_params);
    int variance_result = wait_for_spatz();
    uint32_t variance_cycles = read_cycles() - start;
    const float16 expected_variance =
        reference_variance(input, ELEMENTS, ELEMENTS, expected_mean);

    bool correct = mean_result == 0 && variance_result == 0 &&
        kt_bits(*mean) == kt_bits(expected_mean) &&
        kt_bits(*variance) == kt_bits(expected_variance);

    mean_params->local_elements = ELEMENTS / 2u;
    mean_params->channel_offset = CHANNELS / 2u;
    mean_params->num_groups = 2u;
    mean_params->elements_per_group = ELEMENTS / 2u;
    mean_params->channels_per_group = CHANNELS / 2u;
    variance_params->local_elements = ELEMENTS / 2u;
    variance_params->channel_offset = CHANNELS / 2u;
    variance_params->num_groups = 2u;
    variance_params->elements_per_group = ELEMENTS / 2u;
    variance_params->channels_per_group = CHANNELS / 2u;
    spatz_run_task_with_params(
        GROUP_REDUCE_FP16_SPATZ_TASK, (uint32_t)mean_params);
    mean_result = wait_for_spatz();
    const float16 expected_shard_mean =
        reference_mean(input, ELEMENTS / 2u, ELEMENTS / 2u);
    spatz_run_task_with_params(
        GROUP_CENTERED_REDUCE_FP16_SPATZ_TASK, (uint32_t)variance_params);
    variance_result = wait_for_spatz();
    const float16 expected_shard_variance = reference_variance(
        input, ELEMENTS / 2u, ELEMENTS / 2u, expected_shard_mean);
    correct &= mean_result == 0 && variance_result == 0 &&
        kt_bits(mean[0]) == kt_bits((float16)0.0f) &&
        kt_bits(mean[1]) == kt_bits(expected_shard_mean) &&
        kt_bits(variance[0]) == kt_bits((float16)0.0f) &&
        kt_bits(variance[1]) == kt_bits(expected_shard_variance);
    const bool fast_enough = mean_cycles < MAX_REDUCTION_CYCLES &&
        variance_cycles < MAX_REDUCTION_CYCLES;
    printf("[CV32 (%d)] group-reductions mean_cycles=%u variance_cycles=%u "
           "limit=%u output=%s performance=%s\n",
           get_hartid(), mean_cycles, variance_cycles, MAX_REDUCTION_CYCLES,
           correct ? "SUCCESS" : "FAILED",
           fast_enough ? "SUCCESS" : "FAILED");
    spatz_clk_dis();
    return correct && fast_enough ? 0 : -1;
}
