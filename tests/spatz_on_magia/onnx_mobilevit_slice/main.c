// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/* MobileViT nodes 170 through 179, sharded over every tile in the mesh. */
#include "tile.h"

#include "eventunit.h"
#include "fsync.h"
#include "kernel_test_utils.h"
#include "slice_config.h"
#include "utils/maps_operation_indexing.h"

#include "add_fp16_spatz.h"
#include "conv2dgemm_fp16_spatz.h"
#include "groupnorm_fp16_spatz.h"
#include "mul_fp16_spatz.h"
#include "reducesum_fp16_spatz.h"
#include "relu_fp16_spatz.h"
#include "softmax_fp16_spatz.h"

#define HID get_hartid()
#define QKV_ELEMENTS 16448u
#define QUERY_ELEMENTS 64u
#define REDUCED_ELEMENTS 512u

extern const uint8_t mobilevit_slice_data[];
extern uint32_t _spatz_binary_start;

static float16 norm[SLICE_ELEMENTS] __attribute__((section(".l2_arena"), aligned(4)));
static float16 qkv[QKV_ELEMENTS] __attribute__((section(".l2_arena"), aligned(4)));
static float16 query[QUERY_ELEMENTS] __attribute__((section(".l2_arena"), aligned(4)));
static float16 weighted[SLICE_ELEMENTS] __attribute__((section(".l2_arena"), aligned(4)));
static float16 reduced[REDUCED_ELEMENTS] __attribute__((section(".l2_arena"), aligned(4)));
static float16 activated[SLICE_ELEMENTS] __attribute__((section(".l2_arena"), aligned(4)));
static float16 fused[SLICE_ELEMENTS] __attribute__((section(".l2_arena"), aligned(4)));
static float16 projected[SLICE_ELEMENTS] __attribute__((section(".l2_arena"), aligned(4)));
static float16 outputs[MOBILEVIT_SLICE_TOKENS][SLICE_ELEMENTS]
    __attribute__((section(".l2_arena"), aligned(4)));
static volatile uint32_t window_start __attribute__((section(".l2_arena")));
static volatile uint32_t output_cycles[NUM_HARTS][MOBILEVIT_SLICE_TOKENS]
    __attribute__((section(".l2_arena")));

static inline const float16 *data_at(uint32_t offset)
{
    return (const float16 *)(mobilevit_slice_data + offset);
}

static inline uint32_t read_cycle(void)
{
    uint32_t cycle;
    __asm__ volatile("rdcycle %0" : "=r"(cycle));
    return cycle;
}

static inline void layer_barrier(fsync_controller_t *fsync, eu_controller_t *event_unit)
{
    fsync_sync_global(fsync);
    eu_fsync_wait(event_unit, WFE);
}

static void run_token(uint32_t token, fsync_controller_t *fsync,
                      eu_controller_t *event_unit)
{
    const float16 *input = data_at(SLICE_INPUTS_OFFSET) + token * SLICE_ELEMENTS;
    uint32_t input_shape[4] = {1u, 128u, 4u, 16u};
    uint32_t qkv_shape[4] = {1u, 257u, 4u, 16u};
    uint32_t softmax_shape[4] = {4u, 1u, 1u, 16u};

    MAGIA_groupnorm_fp16_spatz(input, norm, data_at(SLICE_GN_SCALE_OFFSET),
                               data_at(SLICE_GN_BIAS_OFFSET), input_shape, 1u,
                               (float16)1.00135803e-05f);
    layer_barrier(fsync, event_unit);
    MAGIA_conv2dgemm_fp16_spatz(norm, data_at(SLICE_QKV_WEIGHT_OFFSET),
                                data_at(SLICE_QKV_BIAS_OFFSET), qkv, input_shape,
                                qkv_shape, 1u, 1u, 1u, 1u, 0u, 0u, 1u, 1);
    layer_barrier(fsync, event_unit);
    MAGIA_softmax_fp16_spatz(qkv, query, softmax_shape);
    layer_barrier(fsync, event_unit);
    MAGIA_mul_bcast_fp16_spatz(qkv + QUERY_ELEMENTS, query, weighted, 128u, 64u, 0u);
    layer_barrier(fsync, event_unit);
    MAGIA_reducesum_fp16_spatz(weighted, reduced, 512u, 16u, 1u);
    layer_barrier(fsync, event_unit);
    MAGIA_relu_fp16_spatz(qkv + QUERY_ELEMENTS + SLICE_ELEMENTS, activated,
                          SLICE_ELEMENTS);
    layer_barrier(fsync, event_unit);
    MAGIA_mul_bcast_fp16_spatz(activated, reduced, fused, 512u, 16u, 1u);
    layer_barrier(fsync, event_unit);
    MAGIA_conv2dgemm_fp16_spatz(fused, data_at(SLICE_OUT_WEIGHT_OFFSET),
                                data_at(SLICE_OUT_BIAS_OFFSET), projected,
                                input_shape, input_shape, 1u, 1u, 1u, 1u,
                                0u, 0u, 1u, 1);
    layer_barrier(fsync, event_unit);
    MAGIA_add_fp16_spatz(input, projected, outputs[token], SLICE_ELEMENTS);
    output_cycles[HID][token] = read_cycle() - window_start;
    layer_barrier(fsync, event_unit);
}

int main(void)
{
    fsync_config_t fsync_config;
    fsync_controller_t fsync;
    eu_config_t event_config;
    eu_controller_t event_unit;
    uint32_t completion_cycles[MOBILEVIT_SLICE_TOKENS];

    kt_spatz_init((uint32_t)&_spatz_binary_start);
    event_config.hartid = HID;
    event_unit.base = 0;
    event_unit.cfg = &event_config;
    event_unit.api = &eu_api;
    eu_fsync_init(&event_unit, 0);
    fsync_config.hartid = HID;
    fsync.base = 0;
    fsync.cfg = &fsync_config;
    fsync.api = &fsync_api;
    fsync_init(&fsync);

    layer_barrier(&fsync, &event_unit);
    if (HID == 0u)
        window_start = read_cycle();
    layer_barrier(&fsync, &event_unit);

    for (uint32_t token = 0u; token < MOBILEVIT_SLICE_TOKENS; ++token) {
        run_token(token, &fsync, &event_unit);
        if (HID == 0u) {
            uint32_t completion = 0u;
            for (uint32_t tile = 0u; tile < NUM_HARTS; ++tile)
                if (output_cycles[tile][token] > completion)
                    completion = output_cycles[tile][token];
            completion_cycles[token] = completion;
        }
    }

    spatz_clk_dis();
    if (HID == 0u) {
        printf("REFERENCE_TILES count=%u\n", NUM_HARTS);
        for (uint32_t token = 0u; token < MOBILEVIT_SLICE_TOKENS; ++token)
            printf("REFERENCE_COMPLETION token=%u cycle=%u\n", token,
                   completion_cycles[token]);

        const float16 *references = data_at(SLICE_REFERENCES_OFFSET);
        for (uint32_t token = 0u; token < MOBILEVIT_SLICE_TOKENS; ++token) {
            uint32_t mismatches = 0u;
            uint32_t nonfinite = 0u;
            for (uint32_t index = 0u; index < SLICE_ELEMENTS; ++index) {
                uint16_t actual_bits = ((const uint16_t *)outputs[token])[index];
                float actual = maps_operation_f16_to_f32(actual_bits);
                float expected = references[token * SLICE_ELEMENTS + index];
                float difference = actual > expected ? actual - expected : expected - actual;
                float expected_absolute = expected < 0.0f ? -expected : expected;
                if ((actual_bits & 0x7c00u) == 0x7c00u)
                    ++nonfinite;
                if (difference > SLICE_ATOL + SLICE_RTOL * expected_absolute)
                    ++mismatches;
            }
            printf("REFERENCE_VALIDATION token=%u mismatches=%u nonfinite=%u\n",
                   token, mismatches, nonfinite);
        }
    }
    return 0;
}
