// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#pragma once

#include <stdint.h>
#include "tile.h"
#include "mg_event.h"

/**
 * Split `n` items into `parts` contiguous shares as even as possible (the first n % parts get one
 * more) and return share `idx`: its first item in *start and its size in *len (0 if it is empty).
 */
static inline __ALWAYS_INLINE_ void
mg_split(uint32_t n, uint32_t parts, uint32_t idx, uint32_t *start, uint32_t *len)
{
    uint32_t q = n / parts, r = n % parts;

    *start = idx * q + (idx < r ? idx : r);
    *len   = q + (idx < r ? 1 : 0);
}

/**
 * Allocate `size` bytes of this tile's L1 (see l1_alloc()), aligned to `align` bytes (a power of
 * two). Returns NULL if L1 is full.
 */
static inline __ALWAYS_INLINE_ void *mg_l1_alloc_aligned(uint32_t size, uint32_t align)
{
    uint32_t pad = (uint32_t)(-(int32_t)(uintptr_t)l1_alloc(0)) & (align - 1);

    if (pad && !l1_alloc(pad))
        return 0;
    return l1_alloc(size);
}

/** The FP16 value with bit pattern `bits`, and back. */
static inline __ALWAYS_INLINE_ float16 mg_fp16_from_bits(uint16_t bits)
{
    return ((union { uint16_t u; float16 f; }){.u = bits}).f;
}

static inline __ALWAYS_INLINE_ uint16_t mg_fp16_bits(float16 x)
{
    return ((union { float16 f; uint16_t u; }){.f = x}).u;
}
