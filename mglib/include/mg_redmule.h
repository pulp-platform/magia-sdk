// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#pragma once

#include <stdint.h>
#include "eventunit.h"
#include "redmule.h"
#include "mg_event.h"

/**
 * Issue an asynchronous RedMulE GEMM job and stamp `event` with the id needed
 * to later wait for its completion via mg_redmule_wait().
 *
 * The event id is the real hardware-issued job id, obtained by reading the
 * ACQUIRE register (redmule_api.acquire()). If the hardware's job queue
 * (depth 2) is already full, this call blocks - draining completion pulses
 * through `eu`/`mode`, exactly like mg_redmule_wait() does - until a slot
 * frees and a fresh id can be acquired.
 */
extern void mg_redmule_gemm(redmule_controller_t *ctrl,
                             eu_controller_t *eu,
                             eu_wait_mode_t mode,
                             uint32_t x,
                             uint32_t w,
                             uint32_t y,
                             uint16_t m,
                             uint16_t n,
                             uint16_t k,
                             mg_event_t *event,
                             mg_event_callback_t callback);

/**
 * Block (per `mode`) until `event` (as produced by mg_redmule_gemm) has
 * completed, then run its callback if set.
 *
 * RedMulE jobs complete strictly in issue order, so this consumes hardware
 * completion pulses one at a time - advancing a completion counter - until
 * that counter has caught up with `event`'s id. That may happen immediately,
 * if the job already completed while waiting on a later event, or only after
 * spinning on new pulses.
 */
extern void mg_redmule_wait(eu_controller_t *eu, eu_wait_mode_t mode, mg_event_t *event);
