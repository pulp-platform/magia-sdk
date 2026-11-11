// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#pragma once

#include <stdint.h>
#include "eventunit32.h"
#include "regs/tile_ctrl.h"
#include "tile.h"
#include "utils/magia_spatz_utils.h"
#include "mg_event.h"

/*
 * Spatz runs one task at a time and has no job queue: start a task with mg_spatz_run() only after
 * the previous one has been waited with mg_spatz_wait(). Each completion raises one Spatz event in
 * the event unit; an event is pending (id 1) until mg_spatz_wait() has consumed its completion.
 */

/**
 * Set up Spatz on this tile: its event-unit line, its binary (the Spatz tasks linked into the
 * program) and its clock. Only the first call does anything, so a kernel may call it every time.
 */
extern void mg_spatz_init(void);

/**
 * Start Spatz task `task` with the parameter block `params` and mark `event` pending, to wait for
 * it with mg_spatz_wait(). The parameter block must stay valid until the task has completed.
 * `mode` is unused here, and kept for symmetry with mg_spatz_wait().
 */
static inline __ALWAYS_INLINE_ void mg_spatz_run(eu_wait_mode_t mode,
                                                 uint32_t task,
                                                 const void *params,
                                                 mg_event_t *event,
                                                 mg_event_callback_t callback)
{
    (void)mode;
    /* the stores to the parameter block must be issued before Spatz is started */
    asm volatile("" ::: "memory");
    spatz_run_task_with_params(task, (uint32_t)params);
    mg_event_init(event, 1, callback);
}

/**
 * Block (per `mode`) until the task of `event` has completed, then run its callback if set.
 * Waiting again on the same event returns at once.
 */
static inline __ALWAYS_INLINE_ void mg_spatz_wait(eu_wait_mode_t mode, mg_event_t *event)
{
    if (event->id) {
        eu_wait_events(EU_SPATZ_DONE_MASK, mode, 0);
        event->id = 0;
    }
    mg_event_trigger(event);
}

/** Exit code of the last task that completed. */
static inline __ALWAYS_INLINE_ uint32_t mg_spatz_exit_code(void)
{
    return spatz_get_exit_code();
}

/**
 * Run task `task` with `params` and wait for it (WFE): returns its exit code. With MG_SPATZ_CHECK
 * defined, a nonzero exit code is also printed.
 */
static inline __ALWAYS_INLINE_ int mg_spatz_call(uint32_t task, const void *params)
{
    mg_event_t event;

    mg_spatz_run(WFE, task, params, &event, 0);
    mg_spatz_wait(WFE, &event);
    int ret = (int)mg_spatz_exit_code();
#ifdef MG_SPATZ_CHECK
    if (ret)
        printf("[mg_spatz] task 0x%x failed with exit code %d\n", task, ret);
#endif
    return ret;
}
