// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Circular-buffer / multi-producer test for the L1 FIFO (utils/l1_fifo.h). Every
// non-consumer tile is a producer that streams FIFO_RING_MSGS_PER_PROD row-block
// messages into tile 0's FIFO, each producer writing its own sub-ring (keyed by
// hartid). The consumer drains all of them concurrently, round-robin across
// sub-rings. Because each sub-ring holds only FIFO_RING_SLOTS_PER_PROD slots
// (fewer than the messages a producer sends), head/tail wrap the sub-rings
// several times and producers repeatedly block on backpressure — exercising the
// genuine circular behaviour: slot reuse via modular indexing, single-writer
// per-sub-ring head/tail counters, flow control, starvation-free draining, and
// the fill indicator.
//
// Each message is tagged with its producer's hartid and carries its message index
// in the descriptor's dim-0 start; the consumer verifies every (producer, message)
// arrives exactly once with the expected packed payload, and that the live
// occupancy never exceeds the capacity. The consumer uses fifo_peek/fifo_release
// (not fifo_pop) so each payload is read while its slot is still pinned, keeping
// the zero-copy read race-free against the concurrently-pushing producers.
//
// Runs at any tile count >= 2.

#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"
#include "utils/l1_fifo.h"

#define WAIT_MODE WFE

#define NUM_TILES   (MESH_X_TILES * MESH_Y_TILES)
#define NUM_PROD    (NUM_TILES - 1u)
#define BLOCK_ELEMS (FIFO_RING_ROWS * FIFO_RING_COLS)

static inline void poke8(uint32_t addr, uint8_t v) { *(volatile uint8_t *)addr = v; }

// Out-of-line to dodge the -O2 PULP GCC volatile-byte-load miscompile.
static __attribute__((noinline)) uint32_t peek8(uint32_t addr)
{
    return (uint32_t)(*(volatile uint8_t *)addr) & 0xFFu;
}

// Deterministic per-(producer, message, element, byte) pattern.
static inline uint8_t pattern_byte(uint32_t prod, uint32_t m, uint32_t k, uint32_t j)
{
    return (uint8_t)(0x11u + prod * 53u + m * 17u + k * 3u + j);
}

int main(void)
{
    uint32_t hartid = get_hartid();

    idma_config_t idma_cfg      = {.hartid = hartid};
    idma_controller_t idma_ctrl = {.base = NULL, .cfg = &idma_cfg, .api = &idma_api};
    idma_init(&idma_ctrl);

    fsync_config_t fsync_cfg      = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {.base = NULL, .cfg = &fsync_cfg, .api = &fsync_api};
    fsync_init(&fsync_ctrl);

#if STALLING == 0
    eu_config_t eu_cfg      = {.hartid = hartid};
    eu_controller_t eu_ctrl = {.base = NULL, .cfg = &eu_cfg, .api = &eu_api};
    eu_init(&eu_ctrl);
    eu_clear_events(0xFFFFFFFF);
    eu_fsync_init(&eu_ctrl, 0);
    eu_idma_init(&eu_ctrl, 0);
    eu_controller_t *eu = &eu_ctrl;
#else
    eu_controller_t *eu = NULL;
#endif

    // Consumer gives one sub-ring per possible producer (keyed by hartid);
    // producers hold no FIFO of their own.
    uint32_t num_producers = (hartid == FIFO_RING_CONSUMER) ? NUM_TILES : 0u;
    fifo_init(hartid, num_producers, FIFO_RING_SLOTS_PER_PROD, FIFO_RING_SLOT_BYTES);

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    uint32_t errors = 0u;

    // ---- Producers: stream messages into their own sub-ring (backpressured) --
    if (hartid != FIFO_RING_CONSUMER) {
        uint32_t buf = get_l1_base(hartid) + 0x1000u;

        for (uint32_t m = 0; m < FIFO_RING_MSGS_PER_PROD; ++m) {
            // Build a packed ROWS x COLS block with this (producer, message) pattern.
            for (uint32_t k = 0; k < BLOCK_ELEMS; ++k)
                for (uint32_t j = 0; j < FIFO_RING_ELEM_BYTES; ++j)
                    poke8(buf + k * FIFO_RING_ELEM_BYTES + j, pattern_byte(hartid, m, k, j));

            tensor_sub_slice_t src = {
                .rank      = 2u,
                .num_elems = BLOCK_ELEMS,
                .dims      = {{0u, FIFO_RING_ROWS, FIFO_RING_COLS * FIFO_RING_ELEM_BYTES},
                              {0u, FIFO_RING_COLS, FIFO_RING_ELEM_BYTES}},
            };
            // Publish the message index as dim-0 start so the consumer can recover it.
            tensor_sub_slice_t desc = {
                .rank      = 2u,
                .num_elems = BLOCK_ELEMS,
                .dims      = {{m, FIFO_RING_ROWS, FIFO_RING_COLS * FIFO_RING_ELEM_BYTES},
                              {0u, FIFO_RING_COLS, FIFO_RING_ELEM_BYTES}},
            };

            fifo_push_req_t req = {
                .target_hartid = FIFO_RING_CONSUMER,
                .producer_idx  = hartid,
                .src_base_addr = buf,
                .src           = &src,
                .desc          = &desc,
                .tag           = hartid,
                .elem_bytes    = FIFO_RING_ELEM_BYTES,
            };
            if (fifo_push(&idma_ctrl, eu, &req) != 0)
                errors++;
        }
    }

    // ---- Consumer: drain round-robin, verifying payload and occupancy --------
    if (hartid == FIFO_RING_CONSUMER) {
        static uint8_t seen[FIFO_RING_MAX_TILES * FIFO_RING_MSGS_PER_PROD];
        for (uint32_t i = 0; i < NUM_PROD * FIFO_RING_MSGS_PER_PROD; ++i)
            seen[i] = 0u;

        uint32_t max_fill  = 0u;
        uint32_t remaining = NUM_PROD * FIFO_RING_MSGS_PER_PROD;
        while (remaining > 0u) {
            fifo_msg_t msg;
            while (!fifo_peek(hartid, &msg))
                ; // wait for any sub-ring to become non-empty

            // Fill indicator: total occupancy is bounded by the total capacity,
            // and this producer's sub-ring by its own capacity.
            uint32_t fill = fifo_count(hartid);
            if (fill == 0u || fill > fifo_capacity(hartid))
                errors++;
            if (fifo_count_from(hartid, msg.src) > FIFO_RING_SLOTS_PER_PROD)
                errors++;
            if (fill > max_fill)
                max_fill = fill;

            uint32_t prod = msg.tag;                // producer hartid
            uint32_t m    = msg.desc.dims[0].start; // message index

            if (prod < 1u || prod >= NUM_TILES || m >= FIFO_RING_MSGS_PER_PROD || msg.src != prod) {
                printf("[fifo_ring] bad message src=%u tag=%u m=%u\n", msg.src, prod, m);
                errors++;
                fifo_release(hartid, msg.src);
                remaining--;
                continue;
            }

            uint32_t flat = (prod - 1u) * FIFO_RING_MSGS_PER_PROD + m;
            if (seen[flat]) {
                printf("[fifo_ring] duplicate message prod=%u m=%u\n", prod, m);
                errors++;
                fifo_release(hartid, msg.src);
                remaining--;
                continue;
            }
            seen[flat] = 1u;

            if (msg.data_size != BLOCK_ELEMS * FIFO_RING_ELEM_BYTES)
                errors++;

            for (uint32_t k = 0; k < BLOCK_ELEMS; ++k)
                for (uint32_t j = 0; j < FIFO_RING_ELEM_BYTES; ++j) {
                    uint32_t got = peek8(msg.data_ptr + k * FIFO_RING_ELEM_BYTES + j);
                    uint32_t exp = pattern_byte(prod, m, k, j);
                    if (got != exp)
                        errors++;
                }

            // Consume only after the payload has been fully read.
            fifo_release(hartid, msg.src);
            remaining--;
        }

        if (!fifo_is_empty(hartid))
            errors++;

        printf("Tile %u fifo_ring drained %u messages (per-prod cap %u, peak fill %u), errors: %u\n",
               hartid,
               NUM_PROD * FIFO_RING_MSGS_PER_PROD,
               FIFO_RING_SLOTS_PER_PROD,
               max_fill,
               errors);
    }

    fsync_sync_global(&fsync_ctrl);
#if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
#endif

    return errors;
}
