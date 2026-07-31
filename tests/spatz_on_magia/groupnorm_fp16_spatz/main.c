// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * Test for the FP16 groupnorm Spatz kernel in kernels/spatz_fp16/groupnorm.
 *
 * Two cases, see test_data/generator.py: a small one whose 18 shards divide
 * evenly into neither mesh size, and a 65536-element group - large enough that
 * the sum, the sum of squares and the element count itself all leave FP16
 * range, which is what the kernel's scaled reductions exist for.
 *
 * The output buffer is per-tile and pre-filled with FP16 NaN, so kt_check only
 * compares what this tile actually wrote; summing the counts over the tiles is
 * what shows the tiles together covered the tensor. It lives in .l2_arena
 * (NOLOAD, in the L2 below the program image) because at 64 tiles the large
 * case would otherwise put 8 MB of zeroes in the ELF.
 */

#include "tile.h"

#include "eventunit.h"

#include "data.h"
#include "groupnorm_fp16_spatz.h"
#include "kernel_test_utils.h"

#define HID get_hartid()

static float16 Y[NUM_HARTS][OUT_LEN_MAX] __attribute__((section(".l2_arena"), aligned(4)));

/* Defined by the CV32 linker script; the Spatz blob is embedded by the kernel */
extern uint32_t _spatz_binary_start;

static bool run_case(const char *name,
                     uint32_t n,
                     uint32_t c,
                     uint32_t h,
                     uint32_t w,
                     uint32_t num_groups,
                     uint32_t out_len,
                     const float16 *X,
                     const float16 *scale,
                     const float16 *bias,
                     const float16 *golden)
{
    uint32_t in_shape[4] = {n, c, h, w};
    bool ok;

    kt_mark_unwritten(Y[HID], out_len);

    MAGIA_groupnorm_fp16_spatz(X, Y[HID], scale, bias, in_shape, num_groups, (float16)EPSILON);

    ok = kt_check(Y[HID], golden, out_len, ULP_TOLL);
    printf("[CV32 (%d)] %s %s\n", HID, name, ok ? "SUCCESS" : "FAILED");

    return ok;
}

int main(void)
{
    bool ok;

    if (HID == 0)
        printf("\n########## GROUPNORM_FP16_SPATZ KERNEL TEST (%d cases) on %d Tiles "
               "##########\n\n",
               NUM_CASES,
               NUM_HARTS);

    kt_spatz_init((uint32_t)&_spatz_binary_start);

    ok = run_case("2x27x4x4, 9 groups (18 shards)",
                  IN_N_S,
                  IN_C_S,
                  IN_H_S,
                  IN_W_S,
                  NUM_GROUPS_S,
                  OUT_LEN_S,
                  X_S,
                  SCALE_S,
                  BIAS_S,
                  G_S);

    ok &= run_case("1x64x32x32, 1 group (65536 elems/group)",
                   IN_N_L,
                   IN_C_L,
                   IN_H_L,
                   IN_W_L,
                   NUM_GROUPS_L,
                   OUT_LEN_L,
                   X_L,
                   SCALE_L,
                   BIAS_L,
                   G_L);

    spatz_clk_dis();

    printf("[CV32 (%d)] Test %s\n", HID, ok ? "SUCCESS" : "FAILED");

    if (HID == 0)
        printf("\n##########################################################################\n\n");

    return ok ? 0 : -1;
}
