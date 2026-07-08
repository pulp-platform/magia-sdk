// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#pragma once

#include <stdint.h>

#ifndef __ALWAYS_INLINE_
#define __ALWAYS_INLINE_ __attribute__((always_inline))
#endif

/** Forward declaration so the callback signature can reference the event type. */
typedef struct mg_event mg_event_t;

/**
 * @brief Callback invoked when an event is triggered.
 */
typedef void (*mg_event_callback_t)(mg_event_t *event);

/**
 * Generic event record: an identifier plus the callback to run for it.
 */
struct mg_event {
    int32_t id;                   /**< Event identifier. */
    mg_event_callback_t callback; /**< Handler invoked when the event fires. */
};

/**
 * @brief Initialize an event with an id and callback.
 */
static inline __ALWAYS_INLINE_ void
mg_event_init(mg_event_t *event, int32_t id, mg_event_callback_t callback)
{
    event->id       = id;
    event->callback = callback;
}

/**
 * @brief Trigger an event, invoking its callback if set.
 */
static inline __ALWAYS_INLINE_ void mg_event_trigger(mg_event_t *event)
{
    if (event->callback) {
        event->callback(event);
    }
}

/**
 * Circular sequence comparison: true if `a` has occurred at or after `b` on
 * an 8-bit wrapping counter (e.g. a hardware-issued/completed job or transfer
 * counter). Only valid while the two are never more than 127 apart.
 */
static inline int mg_seq_ge(uint8_t a, uint8_t b)
{
    return (int8_t)(a - b) >= 0;
}
