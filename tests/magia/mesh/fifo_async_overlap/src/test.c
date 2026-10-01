// Copyright 2026 Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "eventunit.h"
#include "fsync.h"
#include "idma.h"
#include "tile.h"
#include "utils/l1_fifo.h"
#include "utils/performance_utils.h"
#include "utils/printf.h"

#define PRODUCER_HARTID 0u
#define CONSUMER_HARTID 1u
#define SOURCE_OFFSET 0x00020000u
#define TRANSFER_BYTES 16384u

static tensor_sub_slice_t packed_slice(void)
{
    tensor_sub_slice_t slice = {
        .rank = 1u,
        .num_elems = TRANSFER_BYTES,
        .dims = {{0u, TRANSFER_BYTES, 1u}},
    };
    return slice;
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
        fifo_init(hartid, 2u, 1u, TRANSFER_BYTES);

    fsync_sync_global(&fsync);
    eu_fsync_wait(&eu, WFE);

    uint32_t errors = 0u;
    if (hartid == PRODUCER_HARTID) {
        volatile uint8_t *source =
            (volatile uint8_t *)(get_l1_base(hartid) + SOURCE_OFFSET);
        for (uint32_t producer = 0u; producer < 2u; ++producer)
            for (uint32_t index = 0u; index < TRANSFER_BYTES; ++index)
                source[producer * TRANSFER_BYTES + index] =
                    (uint8_t)((index * 37u + 11u + producer * 73u) & 0xffu);

        tensor_sub_slice_t packed = packed_slice();
        fifo_push_req_t request = {
            .target_hartid = CONSUMER_HARTID,
            .producer_idx = 0u,
            .src_base_addr = (uint32_t)source,
            .src = &packed,
            .desc = &packed,
            .tag = 0xabcdu,
            .elem_bytes = 1u,
        };
        fifo_push_req_t second_request = request;
        second_request.producer_idx = 1u;
        second_request.src_base_addr += TRANSFER_BYTES;
        second_request.tag = 0xbcdeu;
        fifo_pending_push_t pending[2];

        const uint32_t dma_start = perf_get_cycles();
        errors += fifo_push_async_start(&idma, &request, &pending[0]) != 0;
        errors += fifo_push_async_start(&idma, &second_request, &pending[1]) != 0;
        const uint32_t compute_start = perf_get_cycles();

        /* Publication must wait even though the DMA is already running. */
        errors += fifo_count_from(CONSUMER_HARTID, 0u) != 0u;
        const uint32_t overlap_observed =
            !fifo_push_async_is_done(&pending[0]) ||
            !fifo_push_async_is_done(&pending[1]);
        errors += !overlap_observed;

        volatile uint32_t computation = 1u;
        for (uint32_t index = 0u; index < 4096u; ++index)
            computation = computation * 1664525u + 1013904223u;
        const uint32_t compute_end = perf_get_cycles();

        errors += fifo_push_async_finish(&eu, &pending[1]) != 0;
        errors += fifo_push_async_finish(&eu, &pending[0]) != 0;
        const uint32_t dma_done = perf_get_cycles();

        errors += fifo_count_from(CONSUMER_HARTID, 0u) != 1u;
        errors += fifo_count_from(CONSUMER_HARTID, 1u) != 1u;
        errors += !(dma_start < compute_start && compute_start < dma_done);
        errors += !(dma_start < compute_end && computation != 0u);

        printf("fifo_async_overlap dma=[%u,%u] compute=[%u,%u] overlap=%u errors=%u\n",
               dma_start, dma_done, compute_start, compute_end,
               overlap_observed, errors);
    } else if (hartid == CONSUMER_HARTID) {
        uint32_t seen = 0u;
        for (uint32_t message_index = 0u; message_index < 2u; ++message_index) {
            fifo_msg_t message;
            while (!fifo_peek(hartid, &message))
                ;

            uint32_t producer = message.src;
            errors += producer >= 2u;
            if (producer >= 2u)
                producer = 0u;
            errors += (seen & (1u << producer)) != 0u;
            seen |= 1u << producer;
            errors += message.tag != (producer == 0u ? 0xabcdu : 0xbcdeu);
            errors += message.data_size != TRANSFER_BYTES;
            volatile const uint8_t *payload =
                (volatile const uint8_t *)message.data_ptr;
            for (uint32_t index = 0u; index < TRANSFER_BYTES; ++index) {
                errors += payload[index] !=
                    (uint8_t)((index * 37u + 11u + producer * 73u) & 0xffu);
            }
            fifo_release(hartid, message.src);
        }
        errors += seen != 3u;
        printf("fifo_async_overlap consumer errors=%u\n", errors);
    }

    fsync_sync_global(&fsync);
    eu_fsync_wait(&eu, WFE);
    return (int)errors;
}
