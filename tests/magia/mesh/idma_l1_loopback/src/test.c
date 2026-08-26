// Copyright 2026 Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"
#include "utils/maps_utils_v2.h"
#include "utils/performance_utils.h"
#include "utils/printf.h"

#define SOURCE_OFFSET      0x00020000u
#define DESTINATION_OFFSET 0x00050000u
#define MAX_TRANSFER_BYTES 65536u
#define TIMED_REPETITIONS  5u

static void initialize_buffers(uint32_t source_address, uint32_t destination_address)
{
    volatile uint8_t *source      = (volatile uint8_t *)source_address;
    volatile uint8_t *destination = (volatile uint8_t *)destination_address;

    for (uint32_t index = 0u; index < MAX_TRANSFER_BYTES; ++index) {
        source[index]      = (uint8_t)((index * 37u + 19u) & 0xffu);
        destination[index] = 0u;
    }
}

static uint32_t check_copy(uint32_t source_address,
                           uint32_t destination_address,
                           uint32_t size_bytes)
{
    volatile const uint8_t *source = (volatile const uint8_t *)source_address;
    volatile const uint8_t *destination =
        (volatile const uint8_t *)destination_address;
    uint32_t errors = 0u;

    for (uint32_t index = 0u; index < size_bytes; ++index)
        errors += source[index] != destination[index];

    return errors;
}

static uint32_t check_qkv_2d_copy(uint32_t source_address, uint32_t destination_address)
{
    volatile const uint8_t *source = (volatile const uint8_t *)source_address;
    volatile const uint8_t *destination =
        (volatile const uint8_t *)destination_address;
    uint32_t errors = 0u;

    for (uint32_t row = 0u; row < 128u; ++row)
        for (uint32_t byte = 0u; byte < 64u; ++byte)
            errors += source[row * 128u + byte] != destination[row * 64u + byte];

    return errors;
}

static uint32_t check_qkv_fragment_unpack(uint32_t source_address,
                                          uint32_t destination_address)
{
    volatile const uint8_t *source = (volatile const uint8_t *)source_address;
    volatile const uint8_t *destination =
        (volatile const uint8_t *)destination_address;
    uint32_t errors = 0u;

    for (uint32_t channel = 0u; channel < 85u; ++channel)
        for (uint32_t byte = 0u; byte < 64u; ++byte)
            errors += source[channel * 64u + byte] !=
                      destination[channel * 128u + byte];

    return errors;
}

int main(void)
{
    const uint32_t hartid = get_hartid();
    if (hartid != 0u)
        return 0;

    const uint32_t l1_base             = get_l1_base(hartid);
    const uint32_t source_address      = l1_base + SOURCE_OFFSET;
    const uint32_t destination_address = l1_base + DESTINATION_OFFSET;
    const uint32_t sizes[]             = {64u, 256u, 1024u, 8192u, 65536u};

    idma_config_t idma_cfg      = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = 0u,
        .cfg  = &idma_cfg,
        .api  = &idma_api,
    };
    eu_config_t eu_cfg      = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = 0u,
        .cfg  = &eu_cfg,
        .api  = &eu_api,
    };

    idma_init(&idma_ctrl);
    eu_init(&eu_ctrl);
    eu_clear_events(0xffffffffu);
    eu_idma_init(&eu_ctrl, 0u);
    initialize_buffers(source_address, destination_address);

    /* First use polling so an unsupported self-route terminates with a timeout. */
    idma_memcpy_1d(&idma_ctrl, 1u, destination_address, source_address, sizes[0]);
    const uint32_t route_completed = eu_idma_wait_o2a(&eu_ctrl, POLLING);
    const uint32_t route_errors =
        route_completed ? check_copy(source_address, destination_address, sizes[0]) : sizes[0];
    printf("idma_l1_loopback route_completed=%u errors=%u\n", route_completed, route_errors);
    if (route_completed == 0u || route_errors != 0u)
        return 1;

    for (uint32_t size_index = 0u; size_index < sizeof(sizes) / sizeof(sizes[0]);
         ++size_index) {
        const uint32_t size_bytes = sizes[size_index];
        uint32_t total_cycles     = 0u;
        uint32_t minimum_cycles   = UINT32_MAX;

        for (uint32_t repetition = 0u; repetition < TIMED_REPETITIONS; ++repetition) {
            const uint32_t start_cycles = perf_get_cycles();
            idma_memcpy_1d(
                &idma_ctrl, 1u, destination_address, source_address, size_bytes);
            eu_idma_wait_o2a(&eu_ctrl, WFE);
            const uint32_t elapsed_cycles = perf_get_cycles() - start_cycles;
            total_cycles += elapsed_cycles;
            if (elapsed_cycles < minimum_cycles)
                minimum_cycles = elapsed_cycles;
        }

        const uint32_t errors = check_copy(source_address, destination_address, size_bytes);
        printf("idma_l1_loopback bytes=%u avg_cycles=%u min_cycles=%u errors=%u\n",
               size_bytes,
               total_cycles / TIMED_REPETITIONS,
               minimum_cycles,
               errors);
        if (errors != 0u)
            return 1;
    }

    uint32_t qkv_total_cycles   = 0u;
    uint32_t qkv_minimum_cycles = UINT32_MAX;
    for (uint32_t repetition = 0u; repetition < TIMED_REPETITIONS; ++repetition) {
        const uint32_t start_cycles = perf_get_cycles();
        idma_memcpy_2d_ex(&idma_ctrl,
                          1u,
                          destination_address,
                          source_address,
                          64u,
                          64u,
                          128u,
                          128u);
        eu_idma_wait_o2a(&eu_ctrl, WFE);
        const uint32_t elapsed_cycles = perf_get_cycles() - start_cycles;
        qkv_total_cycles += elapsed_cycles;
        if (elapsed_cycles < qkv_minimum_cycles)
            qkv_minimum_cycles = elapsed_cycles;
    }
    const uint32_t qkv_errors = check_qkv_2d_copy(source_address, destination_address);
    printf("idma_l1_loopback qkv_2d_rows=128 row_bytes=64 src_stride=128 "
           "dst_stride=64 avg_cycles=%u min_cycles=%u errors=%u\n",
           qkv_total_cycles / TIMED_REPETITIONS,
           qkv_minimum_cycles,
           qkv_errors);
    if (qkv_errors != 0u)
        return 1;

    initialize_buffers(source_address, destination_address);
    fifo_msg_t message = {
        .data_ptr = source_address,
        .elem_bytes = sizeof(uint16_t),
        .data_size = 85u * 2u * 16u * sizeof(uint16_t),
        .desc = {
            .rank = 4u,
            .num_elems = 85u * 2u * 16u,
            .dims = {
                {0u, 1u, 10880u},
                {0u, 85u, 128u},
                {0u, 2u, 32u},
                {0u, 16u, 2u},
            },
        },
    };
    subslice_desc_t destination = {
        .rank = 4u,
        .shape = {1u, 85u, 2u, 16u},
        .elem_type = ELEM_F16,
        .elem_bytes = sizeof(uint16_t),
        .strides_bytes = {10880u, 128u, 32u, 2u},
    };

    const uint32_t unpack_start = perf_get_cycles();
    maps_fifo_unpack(
        &message, &destination, destination_address, &idma_ctrl, &eu_ctrl);
    const uint32_t unpack_cycles = perf_get_cycles() - unpack_start;
    const uint32_t unpack_errors =
        check_qkv_fragment_unpack(source_address, destination_address);
    printf("idma_l1_loopback maps_fifo_unpack elements=%u cycles=%u errors=%u\n",
           message.desc.num_elems,
           unpack_cycles,
           unpack_errors);
    if (unpack_errors != 0u)
        return 1;

    return 0;
}
