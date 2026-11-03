// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include <stdint.h>
#include "mg_idma.h"

// The id returned by the HAL launch is the hardware job id (NEXT_ID); a full
// hardware job queue stalls the launch itself, so no software backpressure is
// needed here.

void mg_idma_memcpy_1d(eu_wait_mode_t mode,
                       uint8_t dir,
                       uint32_t axi_addr,
                       uint32_t obi_addr,
                       uint32_t len,
                       mg_event_t *event,
                       mg_event_callback_t callback)
{
    (void)mode;
    mg_event_init(event, idma_memcpy_1d(dir, axi_addr, obi_addr, len), callback);
}

void mg_idma_memcpy_2d(eu_wait_mode_t mode,
                       uint8_t dir,
                       uint32_t axi_addr,
                       uint32_t obi_addr,
                       uint32_t len,
                       uint32_t std,
                       uint32_t reps,
                       mg_event_t *event,
                       mg_event_callback_t callback)
{
    (void)mode;
    mg_event_init(event, idma_memcpy_2d(dir, axi_addr, obi_addr, len, std, reps), callback);
}

// mg_idma_wait() is defined as a static inline in mg_idma.h so it folds into
// its call sites (removing the call/return and the one-time cold I-cache miss,
// and letting constant-propagation drop the callback tail where it is NULL).
