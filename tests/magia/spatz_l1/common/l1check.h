// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>
//
// Helpers of the spatz_l1 tests: setup, barrier, operand generation, L2 -> L1 copies, comparison.

#ifndef L1CHECK_H_
#define L1CHECK_H_

#include <stdint.h>

#include "eventunit.h"
#include "fsync.h"
#include "idma.h"
#include "tile.h"
#include "utils/performance_utils.h"

extern uint32_t _spatz_binary_start;

static inline void l1c_init(void)
{
    fsync_init();
    eu_init();
    eu_spatz_init(0);
    eu_fsync_init(0);
    eu_idma_init(0);
    idma_init();
    spatz_init((uint32_t)&_spatz_binary_start);
    perf_init();
}

/* Cycles spent by this tile in the full API calls and in the L1 API calls (L1C_FULL, L1C_L1). */
static uint32_t l1c_cyc_full __attribute__((section(".tile_bss")));
static uint32_t l1c_cyc_l1 __attribute__((section(".tile_bss")));

#define L1C_TIME(acc, ...)                                                                         \
    do {                                                                                           \
        uint32_t t_ = perf_get_cycles();                                                           \
        __VA_ARGS__;                                                                               \
        acc += perf_get_cycles() - t_;                                                             \
    } while (0)
#define L1C_FULL(...) L1C_TIME(l1c_cyc_full, __VA_ARGS__)
#define L1C_L1(...)   L1C_TIME(l1c_cyc_l1, __VA_ARGS__)

static inline void l1c_sync(void)
{
    fsync_sync_global();
    eu_fsync_wait(WFE);
}

/*
 * Fills p[0 .. n - 1] with pseudo-random FP16 values, |x| in [2^(emin - 15), 2^(emax - 14)),
 * negative too if `neg`. Exponent fields emin .. emax must be in 1 .. 30 (normal numbers).
 */
static inline void l1c_fill(uint16_t *p, uint32_t n, uint32_t seed, uint32_t emin, uint32_t emax, int neg)
{
    uint32_t s = seed * 2654435761u + 1;
    for (uint32_t i = 0; i < n; i++) {
        s              = s * 1664525u + 1013904223u;
        uint32_t r     = s >> 8;
        uint32_t e     = emin + (r >> 10) % (emax - emin + 1);
        uint16_t sign  = (neg && (r & 0x8000)) ? 0x8000 : 0;
        p[i]           = sign | (uint16_t)(e << 10) | (uint16_t)(r & 0x3ff);
    }
}

/* Copies n bytes from L2 to L1 (or back) with the iDMA and waits. */
static inline void l1c_copy_in(void *l1, const void *l2, uint32_t n)
{
    idma_memcpy_1d(0, (uint32_t)l2, (uint32_t)l1, n);
    eu_idma_wait_a2o(WFE);
}

/* Bump allocation of n bytes in this tile's L1, 64-byte aligned. Call l1_alloc_init() first. */
static inline void *l1c_alloc(uint32_t n)
{
    uint32_t pad = (64 - ((uint32_t)l1_alloc(0) & 63)) & 63;
    if (pad)
        l1_alloc(pad);
    return l1_alloc(n);
}

/*
 * Compares the n FP16 values of res (L1 API) with ref (full API): returns how many differ by more
 * than `tol` units in the last place, and stores the largest difference in *max_ulp.
 */
static inline uint32_t l1c_compare(const uint16_t *res, const uint16_t *ref, uint32_t n, uint32_t tol,
                                   uint32_t *max_ulp)
{
    uint32_t errors = 0, mx = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint16_t a = ((volatile const uint16_t *)res)[i], b = ((volatile const uint16_t *)ref)[i];
        /* map the sign-magnitude bits to a monotonic integer scale */
        int32_t oa = (a & 0x8000) ? -(int32_t)(a & 0x7fff) : (int32_t)a;
        int32_t ob = (b & 0x8000) ? -(int32_t)(b & 0x7fff) : (int32_t)b;
        uint32_t d = (uint32_t)(oa > ob ? oa - ob : ob - oa);
        if ((a & 0x7c00) == 0x7c00 || (b & 0x7c00) == 0x7c00)
            d = (a == b) ? 0 : 0xffff; /* Inf / NaN: must match exactly */
        if (d > mx)
            mx = d;
        if (d > tol)
            errors++;
    }
    *max_ulp = mx;
    return errors;
}

/* Prints the result line of this tile and returns the errors. */
static inline int l1c_report(const char *name, uint32_t errors, uint32_t max_ulp)
{
    printf("TILE %d: %s full %d l1 %d cycles, max ulp %d, Number of errors: %d\n", get_hartid(), name, l1c_cyc_full,
           l1c_cyc_l1, max_ulp, errors);
    return (int)errors;
}

#endif /* L1CHECK_H_ */
