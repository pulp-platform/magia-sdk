// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * iDMA staging for the kernels/<op>/fp16/spatz kernels.
 *
 * Every one of those kernels works the same way: the caller's tensors stay where they
 * are - typically L2 - each tile copies in only the shard it owns, runs the Spatz task
 * over L1, and copies its shard back out. This wraps the copy in and out so the kernels
 * move it with the iDMA instead of a per-element mmio_fp16 loop.
 *
 * Two things about the event unit are worth knowing before using this:
 *
 *   - kdma_open() must not call eu_init(). That clears EU_CORE_MASK, which would drop
 *     the Spatz completion event the application enabled (see kt_spatz_init). Only
 *     eu_idma_init() is called, and it just ORs its bits in, so repeating it per kernel
 *     invocation is harmless.
 *   - each transfer is waited on before the next is issued, so a completion event
 *     always belongs to the transfer just programmed.
 *
 * The HAL is controller-free, so kdma_t carries no state; it is kept so the kernels
 * read the same as on the controller-based HAL they were written for.
 */

#ifndef KERNELS_DMA_UTILS_H_
#define KERNELS_DMA_UTILS_H_

#include <stdint.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

typedef struct {
    uint8_t unused;
} kdma_t;

static inline void kdma_open(kdma_t *d)
{
    (void)d;
    idma_init();
    eu_idma_init(0);
}

/* Contiguous L2 -> L1. */
static inline void kdma_in(kdma_t *d, uintptr_t l2_addr, uintptr_t l1_addr, uint32_t bytes)
{
    (void)d;
    if (bytes == 0)
        return;

    idma_memcpy_1d(0, (uint32_t)l2_addr, (uint32_t)l1_addr, bytes);
    eu_idma_wait_a2o(WFE);
}

/* Contiguous L1 -> L2. */
static inline void kdma_out(kdma_t *d, uintptr_t l2_addr, uintptr_t l1_addr, uint32_t bytes)
{
    (void)d;
    if (bytes == 0)
        return;

    idma_memcpy_1d(1, (uint32_t)l2_addr, (uint32_t)l1_addr, bytes);
    eu_idma_wait_o2a(WFE);
}

/*
 * Strided L2 -> L1: `reps` rows of `len` bytes, each side advancing by its own stride.
 * Independent strides are what lets a narrow rectangle land inside a wider L1 buffer,
 * which idma_memcpy_2d cannot express (it forces the L1 stride to equal len).
 */
static inline void kdma_in_2d(kdma_t *d,
                              uintptr_t l2_addr,
                              uintptr_t l1_addr,
                              uint32_t len,
                              uint32_t l2_std,
                              uint32_t l1_std,
                              uint32_t reps)
{
    (void)d;
    if (len == 0 || reps == 0)
        return;

    idma_memcpy_2d_ex(0, (uint32_t)l2_addr, (uint32_t)l1_addr, len, l2_std, l1_std, reps);
    eu_idma_wait_a2o(WFE);
}

/* Strided L1 -> L2. */
static inline void kdma_out_2d(kdma_t *d,
                               uintptr_t l2_addr,
                               uintptr_t l1_addr,
                               uint32_t len,
                               uint32_t l2_std,
                               uint32_t l1_std,
                               uint32_t reps)
{
    (void)d;
    if (len == 0 || reps == 0)
        return;

    idma_memcpy_2d_ex(1, (uint32_t)l2_addr, (uint32_t)l1_addr, len, l2_std, l1_std, reps);
    eu_idma_wait_o2a(WFE);
}

#endif /* KERNELS_DMA_UTILS_H_ */
