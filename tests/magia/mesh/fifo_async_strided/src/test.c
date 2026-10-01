// Copyright 2026 Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "eventunit.h"
#include "fsync.h"
#include "idma.h"
#include "tile.h"
#include "utils/l1_fifo.h"
#include "utils/printf.h"

#define PRODUCER_HARTID 0u
#define CONSUMER_HARTID 1u
#define SOURCE_OFFSET 0x00020000u
#define PAGES 8u
#define ROWS 16u
#define COLUMNS 256u
#define ROW_STRIDE 384u
#define PAGE_STRIDE 6272u
#define TRANSFER_BYTES (PAGES * ROWS * COLUMNS)
#define FALLBACK_OFFSET 0x00010000u
#define FALLBACK_ELEMS (2u * 3u * 4u * 5u)

static uint8_t expected_byte(uint32_t page, uint32_t row, uint32_t column)
{
    return (uint8_t)((page * 71u + row * 29u + column * 13u + 5u) & 0xffu);
}

int main(void)
{
    const uint32_t hartid = get_hartid();
    idma_config_t idma_cfg = {.hartid = hartid};
    idma_controller_t idma = {.base = 0u, .cfg = &idma_cfg, .api = &idma_api};
    eu_config_t eu_cfg = {.hartid = hartid};
    eu_controller_t eu = {.base = 0u, .cfg = &eu_cfg, .api = &eu_api};
    fsync_config_t fsync_cfg = {.hartid = hartid};
    fsync_controller_t fsync = {.base = 0u, .cfg = &fsync_cfg, .api = &fsync_api};

    idma_init(&idma);
    eu_init(&eu);
    eu_clear_events(0xffffffffu);
    eu_idma_init(&eu, 0u);
    eu_fsync_init(&eu, 0u);
    fsync_init(&fsync);

    if (hartid == CONSUMER_HARTID)
        fifo_init(hartid, 1u, 1u, TRANSFER_BYTES);
    fsync_sync_global(&fsync);
    eu_fsync_wait(&eu, WFE);

    uint32_t errors = 0u;
    if (hartid == PRODUCER_HARTID) {
        volatile uint8_t *source =
            (volatile uint8_t *)(get_l1_base(hartid) + SOURCE_OFFSET);
        for (uint32_t page = 0u; page < PAGES; ++page)
            for (uint32_t row = 0u; row < ROWS; ++row)
                for (uint32_t column = 0u; column < COLUMNS; ++column)
                    source[page * PAGE_STRIDE + row * ROW_STRIDE + column] =
                        expected_byte(page, row, column);

        tensor_sub_slice_t source_slice = {
            .rank = 4u,
            .num_elems = TRANSFER_BYTES,
            .dims = {{0u, 1u, PAGE_STRIDE * PAGES},
                     {0u, PAGES, PAGE_STRIDE},
                     {0u, ROWS, ROW_STRIDE},
                     {0u, COLUMNS, 1u}},
        };
        tensor_sub_slice_t logical_slice = source_slice;
        fifo_push_req_t request = {
            .target_hartid = CONSUMER_HARTID,
            .producer_idx = 0u,
            .src_base_addr = (uint32_t)source,
            .src = &source_slice,
            .desc = &logical_slice,
            .tag = 0x3d3du,
            .elem_bytes = 1u,
        };
        fifo_pending_push_t pending;
        int result = fifo_push_async_start(&idma, &request, &pending);
        errors += result != 0;
        if (result == 0) {
            errors += fifo_count_from(CONSUMER_HARTID, 0u) != 0u;
            errors += fifo_push_async_finish(&eu, &pending) != 0;
        } else {
            /* Keep the red test finite while recording that async rejected rank-N. */
            errors += fifo_push(&idma, &eu, &request) != 0;
        }
        printf("fifo_async_strided producer errors=%u\n", errors);
    } else if (hartid == CONSUMER_HARTID) {
        fifo_msg_t message;
        while (!fifo_peek(hartid, &message))
            ;
        errors += message.tag != 0x3d3du;
        errors += message.data_size != TRANSFER_BYTES;
        volatile const uint8_t *payload =
            (volatile const uint8_t *)message.data_ptr;
        uint32_t packed = 0u;
        for (uint32_t page = 0u; page < PAGES; ++page)
            for (uint32_t row = 0u; row < ROWS; ++row)
                for (uint32_t column = 0u; column < COLUMNS; ++column)
                    errors += payload[packed++] != expected_byte(page, row, column);
        fifo_release(hartid, message.src);
        printf("fifo_async_strided consumer errors=%u\n", errors);
    }

    fsync_sync_global(&fsync);
    eu_fsync_wait(&eu, WFE);

    /* Four non-unit, non-coalescible dimensions cannot be represented by one
     * hardware descriptor.  Async must report UNSUPPORTED (not queue-full),
     * after which the generalized blocking mover remains the fallback. */
    if (hartid == PRODUCER_HARTID) {
        volatile uint8_t *source = (volatile uint8_t *)(
            get_l1_base(hartid) + SOURCE_OFFSET + FALLBACK_OFFSET);
        for (uint32_t a = 0u; a < 2u; ++a)
            for (uint32_t b = 0u; b < 3u; ++b)
                for (uint32_t c = 0u; c < 4u; ++c)
                    for (uint32_t d = 0u; d < 5u; ++d)
                        source[a * 4096u + b * 768u + c * 96u + d * 2u] =
                            (uint8_t)(a * 67u + b * 23u + c * 7u + d);

        tensor_sub_slice_t source_slice = {
            .rank = 4u,
            .num_elems = FALLBACK_ELEMS,
            .dims = {{0u, 2u, 4096u}, {0u, 3u, 768u},
                     {0u, 4u, 96u}, {0u, 5u, 2u}},
        };
        tensor_sub_slice_t logical_slice = source_slice;
        fifo_push_req_t request = {
            .target_hartid = CONSUMER_HARTID,
            .producer_idx = 0u,
            .src_base_addr = (uint32_t)source,
            .src = &source_slice,
            .desc = &logical_slice,
            .tag = 0x4d4du,
            .elem_bytes = 1u,
        };
        fifo_pending_push_t pending;
        int result = fifo_push_async_start(&idma, &request, &pending);
        errors += result != FIFO_ASYNC_UNSUPPORTED;
        if (result == FIFO_ASYNC_STARTED)
            errors += fifo_push_async_finish(&eu, &pending) != 0;
        else
            errors += fifo_push(&idma, &eu, &request) != 0;
        printf("fifo_async_strided fallback=%d errors=%u\n", result, errors);
    } else if (hartid == CONSUMER_HARTID) {
        fifo_msg_t message;
        while (!fifo_peek(hartid, &message))
            wait_nop(32u);
        errors += message.tag != 0x4d4du;
        errors += message.data_size != FALLBACK_ELEMS;
        volatile const uint8_t *payload =
            (volatile const uint8_t *)message.data_ptr;
        uint32_t packed = 0u;
        for (uint32_t a = 0u; a < 2u; ++a)
            for (uint32_t b = 0u; b < 3u; ++b)
                for (uint32_t c = 0u; c < 4u; ++c)
                    for (uint32_t d = 0u; d < 5u; ++d)
                        errors += payload[packed++] !=
                            (uint8_t)(a * 67u + b * 23u + c * 7u + d);
        fifo_release(hartid, message.src);
        printf("fifo_async_strided fallback consumer errors=%u\n", errors);
    }

    fsync_sync_global(&fsync);
    eu_fsync_wait(&eu, WFE);
    return (int)errors;
}
