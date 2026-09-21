// Copyright 2026 Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"
#include "utils/maps_idma.h"
#include "utils/performance_utils.h"
#include "utils/printf.h"

/* MobileViT nodes 170:179 pattern from idma_nd_issue.pptx.
 *
 * Tile 55 owns a physical 1x128x4x2 F16 tensor and sends one channel as a
 * 1x128x4x1 packed tensor to tile 56. The selected F16 values are four
 * strided rows in each of 128 planes.
 */
#define SOURCE_TILE       55u
#define DESTINATION_TILE  56u
#define SOURCE_OFFSET     0x00020000u
#define DESTINATION_OFFSET 0x00020000u
#define PLANES            128u
#define ROWS_PER_PLANE    4u
#define ELEMENT_BYTES     2u
#define SOURCE_ROW_STRIDE 4u
#define SOURCE_PLANE_STRIDE 16u
#define DESTINATION_ROW_STRIDE 10u
#define DESTINATION_PLANE_STRIDE 40u
#define TIMED_REPETITIONS 5u

static void initialize_pattern(uint32_t source_address, uint32_t destination_address)
{
    volatile uint16_t *source = (volatile uint16_t *)source_address;
    volatile uint16_t *destination = (volatile uint16_t *)destination_address;

    for (uint32_t plane = 0u; plane < PLANES; ++plane) {
        for (uint32_t row = 0u; row < ROWS_PER_PLANE; ++row) {
            const uint32_t source_index = plane * 8u + row * 2u;
            source[source_index] = (uint16_t)(plane * ROWS_PER_PLANE + row);
            source[source_index + 1u] = 0xdeadu;
            destination[(plane * DESTINATION_PLANE_STRIDE +
                         row * DESTINATION_ROW_STRIDE) /
                        ELEMENT_BYTES] = 0xbeefu;
        }
    }
}

static uint32_t check_pattern(uint32_t source_address, uint32_t destination_address)
{
    volatile const uint16_t *source = (volatile const uint16_t *)source_address;
    volatile const uint16_t *destination =
        (volatile const uint16_t *)destination_address;
    uint32_t errors = 0u;

    for (uint32_t plane = 0u; plane < PLANES; ++plane) {
        for (uint32_t row = 0u; row < ROWS_PER_PLANE; ++row) {
            errors += destination[(plane * DESTINATION_PLANE_STRIDE +
                                   row * DESTINATION_ROW_STRIDE) /
                                  ELEMENT_BYTES] !=
                source[plane * 8u + row * 2u];
        }
    }
    return errors;
}

static uint32_t run_1d(
    idma_controller_t *idma, eu_controller_t *event_unit,
    uint32_t source_address, uint32_t destination_address)
{
    const uint32_t start = perf_get_cycles();
    for (uint32_t plane = 0u; plane < PLANES; ++plane) {
        for (uint32_t row = 0u; row < ROWS_PER_PLANE; ++row) {
            idma_memcpy_1d(
                idma,
                1u,
                destination_address + plane * DESTINATION_PLANE_STRIDE +
                    row * DESTINATION_ROW_STRIDE,
                source_address + plane * SOURCE_PLANE_STRIDE +
                    row * SOURCE_ROW_STRIDE,
                ELEMENT_BYTES);
            eu_idma_wait_o2a(event_unit, WFE);
        }
    }
    return perf_get_cycles() - start;
}

static uint32_t run_2d(
    idma_controller_t *idma, eu_controller_t *event_unit,
    uint32_t source_address, uint32_t destination_address)
{
    const uint32_t start = perf_get_cycles();
    for (uint32_t plane = 0u; plane < PLANES; ++plane) {
        idma_memcpy_2d_ex(
            idma,
            1u,
            destination_address + plane * DESTINATION_PLANE_STRIDE,
            source_address + plane * SOURCE_PLANE_STRIDE,
            ELEMENT_BYTES,
            DESTINATION_ROW_STRIDE,
            SOURCE_ROW_STRIDE,
            ROWS_PER_PLANE);
        eu_idma_wait_o2a(event_unit, WFE);
    }
    return perf_get_cycles() - start;
}

static uint32_t run_3d(
    idma_controller_t *idma, eu_controller_t *event_unit,
    uint32_t source_address, uint32_t destination_address)
{
    const uint32_t start = perf_get_cycles();
    idma_memcpy_3d(
        idma,
        1u,
        destination_address,
        source_address,
        ELEMENT_BYTES,
        DESTINATION_ROW_STRIDE,
        SOURCE_ROW_STRIDE,
        ROWS_PER_PLANE,
        DESTINATION_PLANE_STRIDE,
        SOURCE_PLANE_STRIDE,
        PLANES);
    eu_idma_wait_o2a(event_unit, WFE);
    return perf_get_cycles() - start;
}

static uint32_t run_maps(
    idma_controller_t *idma, eu_controller_t *event_unit,
    uint32_t source_address, uint32_t destination_address)
{
    const TensorRange source_ranges[] = {
        {0u, 1u, 2048u}, {0u, 128u, 16u}, {0u, 4u, 4u}, {0u, 1u, 2u}};
    const TensorRange destination_ranges[] = {
        {0u, 1u, 5120u}, {0u, 128u, 40u}, {0u, 4u, 10u}, {0u, 1u, 2u}};
    const tensor_sub_slice_t source = {
        .rank = 4u,
        .num_elems = PLANES * ROWS_PER_PLANE,
        .dims = {
            source_ranges[0], source_ranges[1], source_ranges[2], source_ranges[3]},
    };
    const tensor_sub_slice_t destination = {
        .rank = 4u,
        .num_elems = PLANES * ROWS_PER_PLANE,
        .dims = {destination_ranges[0], destination_ranges[1],
                 destination_ranges[2], destination_ranges[3]},
    };

    const uint32_t start = perf_get_cycles();
    idma_memcpy_md_to_nd(
        idma, 1u, destination_address, source_address,
        &source, &destination, ELEMENT_BYTES, event_unit);
    return perf_get_cycles() - start;
}

int main(void)
{
    const uint32_t hartid = get_hartid();
    if (hartid != SOURCE_TILE)
        return 0;

    const uint32_t source_address = get_l1_base(SOURCE_TILE) + SOURCE_OFFSET;
    const uint32_t destination_address =
        get_l1_base(DESTINATION_TILE) + DESTINATION_OFFSET;

    idma_config_t idma_config = {.hartid = hartid};
    idma_controller_t idma = {.base = 0u, .cfg = &idma_config, .api = &idma_api};
    eu_config_t event_config = {.hartid = hartid};
    eu_controller_t event_unit = {
        .base = 0u, .cfg = &event_config, .api = &eu_api};
    idma_init(&idma);
    eu_init(&event_unit);
    eu_clear_events(0xffffffffu);
    eu_idma_init(&event_unit, 0u);

    initialize_pattern(source_address, destination_address);
    run_1d(&idma, &event_unit, source_address, destination_address);
    uint32_t errors_1d = check_pattern(source_address, destination_address);
    run_2d(&idma, &event_unit, source_address, destination_address);
    uint32_t errors_2d = check_pattern(source_address, destination_address);
    run_3d(&idma, &event_unit, source_address, destination_address);
    uint32_t errors_3d = check_pattern(source_address, destination_address);
    run_maps(&idma, &event_unit, source_address, destination_address);
    uint32_t errors_maps = check_pattern(source_address, destination_address);

    uint32_t total_1d = 0u;
    uint32_t total_2d = 0u;
    uint32_t total_3d = 0u;
    uint32_t total_maps = 0u;
    uint32_t minimum_1d = UINT32_MAX;
    uint32_t minimum_2d = UINT32_MAX;
    uint32_t minimum_3d = UINT32_MAX;
    uint32_t minimum_maps = UINT32_MAX;
    for (uint32_t repetition = 0u; repetition < TIMED_REPETITIONS; ++repetition) {
        const uint32_t cycles_1d =
            run_1d(&idma, &event_unit, source_address, destination_address);
        const uint32_t cycles_2d =
            run_2d(&idma, &event_unit, source_address, destination_address);
        const uint32_t cycles_3d =
            run_3d(&idma, &event_unit, source_address, destination_address);
        const uint32_t cycles_maps =
            run_maps(&idma, &event_unit, source_address, destination_address);
        total_1d += cycles_1d;
        total_2d += cycles_2d;
        total_3d += cycles_3d;
        total_maps += cycles_maps;
        if (cycles_1d < minimum_1d)
            minimum_1d = cycles_1d;
        if (cycles_2d < minimum_2d)
            minimum_2d = cycles_2d;
        if (cycles_3d < minimum_3d)
            minimum_3d = cycles_3d;
        if (cycles_maps < minimum_maps)
            minimum_maps = cycles_maps;
    }

    const uint32_t average_1d = total_1d / TIMED_REPETITIONS;
    const uint32_t average_2d = total_2d / TIMED_REPETITIONS;
    const uint32_t average_3d = total_3d / TIMED_REPETITIONS;
    const uint32_t average_maps = total_maps / TIMED_REPETITIONS;
    const uint32_t speedup_x100 = average_3d == 0u
        ? 0u : (average_2d * 100u) / average_3d;
    const uint32_t errors = errors_1d + errors_2d + errors_3d + errors_maps;
    printf("idma_3d_benchmark pattern=1x128x4x1_f16 source_tile=%u destination_tile=%u\n",
           SOURCE_TILE, DESTINATION_TILE);
    printf("idma_3d_benchmark 1d_descriptors=%u avg_cycles=%u min_cycles=%u\n",
           PLANES * ROWS_PER_PLANE, average_1d, minimum_1d);
    printf("idma_3d_benchmark 2d_descriptors=%u avg_cycles=%u min_cycles=%u\n",
           PLANES, average_2d, minimum_2d);
    printf("idma_3d_benchmark 3d_descriptors=1 avg_cycles=%u min_cycles=%u\n",
           average_3d, minimum_3d);
    printf("idma_3d_benchmark maps_selected=3d avg_cycles=%u min_cycles=%u\n",
           average_maps, minimum_maps);
    printf("idma_3d_benchmark speedup_x100=%u errors=%u\n", speedup_x100, errors);

    return errors != 0u || average_3d >= average_2d || average_maps >= average_2d;
}
