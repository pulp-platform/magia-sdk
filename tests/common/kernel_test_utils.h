// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * Shared harness for the kernels/spatz_fp16/<op> tests.
 *
 * Every one of those kernels splits one dimension over the tiles and writes
 * only its own share of the output, at global indices. Rather than duplicating
 * each kernel's shard arithmetic in its test, every tile gets a private output
 * buffer pre-filled with FP16 NaN: whatever is no longer NaN afterwards is
 * exactly what this tile wrote, and only that is compared against the golden
 * model. Each tile also reports how many elements it wrote, so that summing
 * those counts over the tiles shows whether the tiles together covered the
 * whole tensor - a kernel that silently drops a remainder shard shows up there.
 */

#ifndef KERNEL_TEST_UTILS_H_
#define KERNEL_TEST_UTILS_H_

#include <stdbool.h>
#include <stdint.h>

#include "tile.h"

#include "eventunit.h"

/* FP16 quiet NaN. None of these kernels can produce NaN from the generated
 * inputs, so it doubles as "this element was never written". */
#define KT_UNWRITTEN  (0x7E00u)

/* Mismatching elements printed per tile before the report is truncated */
#define KT_MAX_REPORT (8)

static inline uint16_t kt_bits(float16 x)
{
    uint16_t bits;

    __builtin_memcpy(&bits, &x, sizeof(bits));

    return bits;
}

static inline float16 kt_from_bits(uint16_t bits)
{
    float16 x;

    __builtin_memcpy(&x, &bits, sizeof(bits));

    return x;
}

/* NaN or (+/-)Inf: exponent all ones */
static inline bool kt_is_invalid(uint16_t x)
{
    return (x & 0x7C00) == 0x7C00;
}

/* FP16 bits reordered into a monotonic integer, so that the distance between
 * two of them is the number of representable values in between (1 ULP each) */
static inline int32_t kt_ordered(uint16_t x)
{
    int32_t i = (int32_t)x;

    return (i & 0x8000) ? (0x8000 - (i & 0x7FFF)) : (i + 0x8000);
}

static inline void kt_mark_unwritten(float16 *buf, uint32_t len)
{
    for (uint32_t i = 0; i < len; i++)
        buf[i] = kt_from_bits(KT_UNWRITTEN);
}

/*
 * Compares the elements this tile wrote against the golden model.
 *
 * ulp_toll is 0 for kernels that must be bit-exact (element-wise ops, max,
 * data movement, and reductions whose golden model accumulates in the same
 * order) and a small budget for the ones going through an approximated
 * exp/rsqrt or a different accumulation order.
 *
 * Returns true if every written element is within tolerance. Writing nothing
 * is not a failure per se - with more tiles than shards some tiles get no
 * work - but it is reported so the coverage sum can be checked.
 */
static inline bool
kt_check(const float16 *result, const float16 *golden, uint32_t len, int32_t ulp_toll)
{
    uint32_t written;
    uint32_t errors;
    int32_t ulp_max;
    int32_t ulp_avg;
    int32_t ulp_sum;

    written = 0;
    errors  = 0;
    ulp_max = 0;
    ulp_sum = 0;

    for (uint32_t i = 0; i < len; i++) {
        uint16_t res = kt_bits(result[i]);
        uint16_t exp;
        int32_t ulp;

        if (res == KT_UNWRITTEN)
            continue;

        written++;
        exp = kt_bits(golden[i]);

        /* A NaN/Inf on either side is a failure, not a distance */
        if (kt_is_invalid(exp) || kt_is_invalid(res)) {
            errors++;
            if (errors <= KT_MAX_REPORT)
                printf("[CV32 (%d)] Invalid FP16 at index %d\t-\texpected: %x\t-\tcomputed: %x\n",
                       get_hartid(),
                       i,
                       exp,
                       res);
            continue;
        }

        ulp = kt_ordered(exp) - kt_ordered(res);
        if (ulp < 0)
            ulp = -ulp;

        ulp_sum += ulp;
        if (ulp > ulp_max)
            ulp_max = ulp;

        if (ulp > ulp_toll) {
            errors++;
            if (errors <= KT_MAX_REPORT)
                printf("[CV32 (%d)] Mismatch at index %d\t-\texpected: %x\t-\tcomputed: %x\t-\t"
                       "ulp: %d\n",
                       get_hartid(),
                       i,
                       exp,
                       res,
                       ulp);
        }
    }

    if (errors > KT_MAX_REPORT)
        printf("[CV32 (%d)] ... %d more bad elements\n", get_hartid(), errors - KT_MAX_REPORT);

    ulp_avg = written ? (ulp_sum / (int32_t)written) : 0;

    /* Parsed by the run script to check the tiles covered the whole output */
    printf("[CV32 (%d)] wrote %d/%d - errors %d - ulp max %d avg %d - toll %d\n",
           get_hartid(),
           written,
           len,
           errors,
           ulp_max,
           ulp_avg,
           ulp_toll);

    return errors == 0;
}

/* Brings up the event unit and Spatz. The kernels expect this to have been
 * done by the application: they only run the task and wait for it. */
static inline void kt_spatz_init(uint32_t spatz_binary_start)
{
    eu_config_t eu_cfg;
    eu_controller_t eu_ctrl;

    eu_cfg.hartid = get_hartid();
    eu_ctrl.base  = 0;
    eu_ctrl.cfg   = &eu_cfg;
    eu_ctrl.api   = &eu_api;

    eu_init(&eu_ctrl);
    eu_spatz_init(&eu_ctrl, 0);

    spatz_init(spatz_binary_start);
}

#endif /* KERNEL_TEST_UTILS_H_ */
