// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#pragma once

#include <stdint.h>
#include "eventunit32.h"
#include "redmule.h"
#include "redmule16.h"
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
extern void mg_redmule_gemm(eu_wait_mode_t mode,
                            uint32_t x,
                            uint32_t w,
                            uint32_t y,
                            uint16_t m,
                            uint16_t n,
                            uint16_t k,
                            mg_event_t *event,
                            mg_event_callback_t callback);
extern void mg_redmule_gemm_enqueue(eu_wait_mode_t mode,
                                    uint32_t x,
                                    uint32_t w,
                                    uint32_t y,
                                    uint16_t m,
                                    uint16_t n,
                                    uint16_t k,
                                    mg_event_t *event,
                                    mg_event_callback_t callback);
#if REDMULE_MM != 0
/**
 * Same as mg_redmule_gemm / mg_redmule_gemm_enqueue, but for a MATMUL:
 * y = x @ w, with no accumulation. The previous content of `y` is not read, so
 * it needs no initialization. Jobs of both kinds share the hardware queue: they
 * are committed, started and waited with the same calls.
 *
 * Only available with the memory-mapped HWPE registers (REDMULE_MM=1).
 */
extern void mg_redmule_matmul(eu_wait_mode_t mode,
                              uint32_t x,
                              uint32_t w,
                              uint32_t y,
                              uint16_t m,
                              uint16_t n,
                              uint16_t k,
                              mg_event_t *event,
                              mg_event_callback_t callback);
extern void mg_redmule_matmul_enqueue(eu_wait_mode_t mode,
                                      uint32_t x,
                                      uint32_t w,
                                      uint32_t y,
                                      uint16_t m,
                                      uint16_t n,
                                      uint16_t k,
                                      mg_event_t *event,
                                      mg_event_callback_t callback);
#endif
extern void mg_redmule_gemm_commit_start();
extern void mg_redmule_gemm_start();
extern void mg_redmule_gemm_commit();

/**
 * Block (per `mode`) until `event` (as produced by mg_redmule_gemm) has
 * completed, then run its callback if set.
 *
 * RedMulE jobs complete strictly in issue order, so the job is done once
 * RUNNING_JOB (the count of finished jobs) has caught up with `event`'s id.
 * That may hold immediately, if the job already completed while waiting on a
 * later event; otherwise this sleeps on the completion event and reads
 * RUNNING_JOB again. Completions are counted
 * only from RUNNING_JOB, never from the pulses: a pulse left latched by a job
 * already counted would otherwise count a job still running. A stale pulse
 * only costs one more read, and a completion between the read and the sleep
 * leaves its pulse latched, so it is not lost.
 *
 * Defined here as a static inline (rather than out-of-line in mg_redmule.c) so
 * it folds into its call sites under -O/-flto.
 */
static inline __ALWAYS_INLINE_ void mg_redmule_wait(eu_wait_mode_t mode, mg_event_t *event)
{
    uint8_t target = (uint8_t)(event->id + 1);

    // RUNNING_JOB and the job ids wrap at 8 bits: compared modulo 256 (see mg_seq_ge())
    while (!mg_seq_ge((uint8_t)HWPE_READ(REDMULE_REG_OFFS + REDMULE_RUNNING_JOB), target))
        eu32_redmule_wait(mode);

    mg_event_trigger(event);
}
