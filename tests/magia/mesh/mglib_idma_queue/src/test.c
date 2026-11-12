// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include <stdint.h>

#include "tile.h"
#include "mg_idma.h"
#include "mg_event.h"
#include "fsync.h"

#define WAIT_MODE WFE

// More jobs per direction than the hardware job queue holds (16), so that
// issuing them back to back fills the queue and stalls the launch.
#define N_JOBS    24
#define JOB_BYTES 4096
#define BUF_BYTES (N_JOBS * JOB_BYTES)
#define N_TILES   (MESH_X_TILES * MESH_Y_TILES)

// One L2 slice per tile (.bss is in the shared L2, cleared by hart 0 in crt0).
static uint32_t l2_buf[N_TILES][BUF_BYTES / 4];

// Per-tile callback counter.
static uint32_t n_callbacks __attribute__((section(".tile_bss")));

static void count_callback(mg_event_t *event)
{
    (void)event;
    n_callbacks++;
}

static inline uint32_t pattern(uint32_t hartid, uint32_t i)
{
    return (hartid << 24) ^ (i * 0x9E3779B1u);
}

// Order of the waits: the middle job first, then the last one, then the
// others from the last down, so that most waits find their job already done.
static inline uint32_t wait_order(uint32_t k)
{
    if (k == 0)
        return N_JOBS / 2;
    if (k == 1)
        return N_JOBS - 1;
    uint32_t j = N_JOBS - k; // N_JOBS-2 .. 1, then 0
    return (j <= N_JOBS / 2) ? j - 1 : j;
}

/**
 * Checks the hardware job queue of the iDMA through mglib: on each direction,
 * N_JOBS > 16 transfers are issued back to back and then waited out of order.
 * L1 buffer A -> L2 (O2A) -> L1 buffer B (A2O) must reproduce buffer A, the
 * event ids must be consecutive hardware job ids and every callback must run.
 * The jobs are large enough for the queue to fill while they are issued.
 */
int main(void)
{
    uint32_t hartid = get_hartid();

    idma_init();
    fsync_init();
    eu_init();
    eu_fsync_init(0);
    eu_idma_init(0);

    uint32_t l1_a = get_l1_base(hartid);
    uint32_t l1_b = l1_a + BUF_BYTES;
    uint32_t l2   = (uint32_t)l2_buf[hartid];

    for (uint32_t i = 0; i < BUF_BYTES / 4; i++) {
        ((volatile uint32_t *)l1_a)[i] = pattern(hartid, i);
        ((volatile uint32_t *)l1_b)[i] = 0;
    }

    // hart 0 clears the shared .bss (l2_buf) in crt0 while the other tiles
    // already run: wait for all tiles before the first transfer to L2
    fsync_sync_level(MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(WAIT_MODE);

    uint32_t errors = 0;
    mg_event_t evt[N_JOBS];
    uint32_t max_pending[2] = {0, 0};

    for (uint8_t dir = 1;; dir = 0) {
        uint32_t src = dir ? l1_a : l2;
        uint32_t dst = dir ? l2 : l1_b;

        for (uint32_t j = 0; j < N_JOBS; j++) {
            if (dir)
                mg_idma_memcpy_1d(WAIT_MODE,
                                  1,
                                  dst + j * JOB_BYTES,
                                  src + j * JOB_BYTES,
                                  JOB_BYTES,
                                  &evt[j],
                                  count_callback);
            else
                mg_idma_memcpy_1d(WAIT_MODE,
                                  0,
                                  src + j * JOB_BYTES,
                                  dst + j * JOB_BYTES,
                                  JOB_BYTES,
                                  &evt[j],
                                  count_callback);
        }
        max_pending[dir] = (uint32_t)evt[N_JOBS - 1].id - idma_done_id(dir);

        for (uint32_t j = 1; j < N_JOBS; j++) {
            if ((uint32_t)evt[j].id != (uint32_t)evt[j - 1].id + 1) {
                printf("dir %d: job %d has id %d after %d\n", dir, j, evt[j].id, evt[j - 1].id);
                errors++;
            }
        }

        for (uint32_t k = 0; k < N_JOBS; k++) {
            uint32_t j = wait_order(k);
            mg_idma_wait(dir, WAIT_MODE, &evt[j]);
            // job j is done: its chunk must be in place already
            for (uint32_t i = j * JOB_BYTES / 4; i < (j + 1) * JOB_BYTES / 4; i++) {
                if (((volatile uint32_t *)dst)[i] != pattern(hartid, i)) {
                    printf("dir %d, job %d: word %d is %x instead of %x\n",
                           dir,
                           j,
                           i,
                           ((volatile uint32_t *)dst)[i],
                           pattern(hartid, i));
                    errors++;
                    break;
                }
            }
        }

        if (dir == 0)
            break;
    }

    if (n_callbacks != 2 * N_JOBS) {
        printf("%d callbacks instead of %d\n", n_callbacks, 2 * N_JOBS);
        errors++;
    }

    printf("Jobs pending after the issue: O2A %d, A2O %d. Number of errors: %d\n",
           max_pending[1],
           max_pending[0],
           errors);

    return errors;
}
