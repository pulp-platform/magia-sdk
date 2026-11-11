// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#pragma once

#include <stdint.h>
#include "fsync.h"
#include "eventunit.h"
#include "mg_event.h"

/*
 * Barriers over the tiles of this tile's mesh row, mesh column or the whole mesh. A barrier is
 * started with mg_fsync_<scope>_start() and completed with mg_fsync_wait(); mg_fsync_<scope>()
 * does both. Only one barrier may be in flight at a time.
 */

/** Set up the barriers on this tile: the fsync controller and its event-unit line. */
static inline __ALWAYS_INLINE_ void mg_fsync_init(void)
{
    fsync_init();
    eu_fsync_init(0);
}

static inline __ALWAYS_INLINE_ void mg_fsync_row_start(mg_event_t *event, mg_event_callback_t callback)
{
    fsync_sync_row();
    mg_event_init(event, 0, callback);
}

static inline __ALWAYS_INLINE_ void mg_fsync_col_start(mg_event_t *event, mg_event_callback_t callback)
{
    fsync_sync_col();
    mg_event_init(event, 0, callback);
}

static inline __ALWAYS_INLINE_ void mg_fsync_global_start(mg_event_t *event, mg_event_callback_t callback)
{
    fsync_sync_global();
    mg_event_init(event, 0, callback);
}

/** Block (per `mode`) until the barrier of `event` has completed, then run its callback if set. */
static inline __ALWAYS_INLINE_ void mg_fsync_wait(eu_wait_mode_t mode, mg_event_t *event)
{
    eu_fsync_wait(mode);
    mg_event_trigger(event);
}

static inline __ALWAYS_INLINE_ void mg_fsync_row(eu_wait_mode_t mode)
{
    fsync_sync_row();
    eu_fsync_wait(mode);
}

static inline __ALWAYS_INLINE_ void mg_fsync_col(eu_wait_mode_t mode)
{
    fsync_sync_col();
    eu_fsync_wait(mode);
}

static inline __ALWAYS_INLINE_ void mg_fsync_global(eu_wait_mode_t mode)
{
    fsync_sync_global();
    eu_fsync_wait(mode);
}
