// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * iDMA staging for the kernels/spatz_fp16 kernels.
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
 *   - the GVSoC iDMA controller faults if a transfer is programmed while the previous
 *     one is still in flight, so each transfer is waited on before the next is issued.
 *     There is no queueing to exploit here.
 */

#ifndef KERNELS_DMA_UTILS_H_
#define KERNELS_DMA_UTILS_H_

#include <stdint.h>

#include "eventunit.h"
#include "idma.h"
#include "tile.h"

typedef struct {
    idma_controller_t idma;
    idma_config_t idma_cfg;
    eu_controller_t eu;
    eu_config_t eu_cfg;
} kdma_t;

static inline void kdma_open(kdma_t *d)
{
    d->idma_cfg.hartid = get_hartid();
    d->idma.base       = 0;
    d->idma.cfg        = &d->idma_cfg;
    d->idma.api        = &idma_api;

    d->eu_cfg.hartid = get_hartid();
    d->eu.base       = 0;
    d->eu.cfg        = &d->eu_cfg;
    d->eu.api        = &eu_api;

    idma_init(&d->idma);
    eu_idma_init(&d->eu, 0);
}

/* Contiguous L2 -> L1. */
static inline void kdma_in(kdma_t *d, uintptr_t l2_addr, uintptr_t l1_addr, uint32_t bytes)
{
    if (bytes == 0)
        return;

    idma_memcpy_1d(&d->idma, 0, (uint32_t)l2_addr, (uint32_t)l1_addr, bytes);
    eu_idma_wait_a2o(&d->eu, WFE);
}

/* Contiguous L1 -> L2. */
static inline void kdma_out(kdma_t *d, uintptr_t l2_addr, uintptr_t l1_addr, uint32_t bytes)
{
    if (bytes == 0)
        return;

    idma_memcpy_1d(&d->idma, 1, (uint32_t)l2_addr, (uint32_t)l1_addr, bytes);
    eu_idma_wait_o2a(&d->eu, WFE);
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
    if (len == 0 || reps == 0)
        return;

    idma_memcpy_2d_ex(&d->idma, 0, (uint32_t)l2_addr, (uint32_t)l1_addr, len, l2_std, l1_std, reps);
    eu_idma_wait_a2o(&d->eu, WFE);
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
    if (len == 0 || reps == 0)
        return;

    idma_memcpy_2d_ex(&d->idma, 1, (uint32_t)l2_addr, (uint32_t)l1_addr, len, l2_std, l1_std, reps);
    eu_idma_wait_o2a(&d->eu, WFE);
}

#endif /* KERNELS_DMA_UTILS_H_ */
