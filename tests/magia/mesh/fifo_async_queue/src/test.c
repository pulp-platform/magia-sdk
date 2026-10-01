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
#define NUM_TRANSFERS 17u
#define TRANSFER_BYTES 8192u

static tensor_sub_slice_t packed_slice(void)
{
    tensor_sub_slice_t slice = {
        .rank = 1u,
        .num_elems = TRANSFER_BYTES,
        .dims = {{0u, TRANSFER_BYTES, 1u}},
    };
    return slice;
}

static uint8_t expected_byte(uint32_t transfer, uint32_t index)
{
    return (uint8_t)((index * 29u + transfer * 61u + 7u) & 0xffu);
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
        fifo_init(hartid, NUM_TRANSFERS, 1u, TRANSFER_BYTES);

    fsync_sync_global(&fsync);
    eu_fsync_wait(&eu, WFE);

    uint32_t errors = 0u;
    if (hartid == PRODUCER_HARTID) {
        volatile uint8_t *source =
            (volatile uint8_t *)(get_l1_base(hartid) + SOURCE_OFFSET);
        for (uint32_t transfer = 0u; transfer < NUM_TRANSFERS; ++transfer)
            for (uint32_t index = 0u; index < TRANSFER_BYTES; ++index)
                source[transfer * TRANSFER_BYTES + index] =
                    expected_byte(transfer, index);

        tensor_sub_slice_t packed = packed_slice();
        fifo_pending_push_t pending[NUM_TRANSFERS] = {0};
        uint32_t next_to_finish = 0u;
        uint32_t saturation_count = 0u;

        for (uint32_t transfer = 0u; transfer < NUM_TRANSFERS; ++transfer) {
            fifo_push_req_t request = {
                .target_hartid = CONSUMER_HARTID,
                .producer_idx = transfer,
                .src_base_addr = (uint32_t)&source[transfer * TRANSFER_BYTES],
                .src = &packed,
                .desc = &packed,
                .tag = 0x51000000u + transfer,
                .elem_bytes = 1u,
            };

            for (;;) {
                int rc = fifo_push_async_start(
                    &idma, &request, &pending[transfer]);
                if (rc == FIFO_ASYNC_STARTED)
                    break;
                if (rc != FIFO_ASYNC_IDMA_FULL) {
                    ++errors;
                    break;
                }

                ++saturation_count;
                while (next_to_finish < transfer &&
                       !pending[next_to_finish].active)
                    ++next_to_finish;
                if (next_to_finish == transfer) {
                    ++errors;
                    break;
                }
                errors += fifo_push_async_finish(
                    &eu, &pending[next_to_finish]) != 0;
                ++next_to_finish;
            }
        }

        for (; next_to_finish < NUM_TRANSFERS; ++next_to_finish)
            if (pending[next_to_finish].active)
                errors += fifo_push_async_finish(
                    &eu, &pending[next_to_finish]) != 0;

        errors += saturation_count == 0u;
        printf("fifo_async_queue saturation=%u errors=%u\n",
               saturation_count, errors);
    } else if (hartid == CONSUMER_HARTID) {
        for (uint32_t transfer = 0u; transfer < NUM_TRANSFERS; ++transfer) {
            fifo_msg_t message;
            while (!fifo_peek(hartid, &message))
                wait_nop(32u);

            errors += message.src != transfer;
            errors += message.tag != 0x51000000u + transfer;
            errors += message.data_size != TRANSFER_BYTES;
            volatile const uint8_t *payload =
                (volatile const uint8_t *)message.data_ptr;
            for (uint32_t index = 0u; index < TRANSFER_BYTES; ++index)
                errors += payload[index] != expected_byte(transfer, index);
            fifo_release(hartid, message.src);
        }
        printf("fifo_async_queue consumer errors=%u\n", errors);
    }

    fsync_sync_global(&fsync);
    eu_fsync_wait(&eu, WFE);
    return (int)errors;
}
