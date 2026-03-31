// Copyright 2026 Fondazione ChipsIT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alex Marchioni <alex.marchioni@chips.it>

#ifndef _TEST_INCLUDE_GUARD_
#define _TEST_INCLUDE_GUARD_

#include <stdint.h>

#define INPUTS_NUM 2

#define INPUT0_NDIM 4
#define INPUT0_DIM0 1
#define INPUT0_DIM1 5
#define INPUT0_DIM2 5
#define INPUT0_DIM3 5
#define INPUT0_SIZE 125
#define INPUT0_TYPE int8_t

#define INPUT1_NDIM 4
#define INPUT1_DIM0 1
#define INPUT1_DIM1 5
#define INPUT1_DIM2 5
#define INPUT1_DIM3 5
#define INPUT1_SIZE 125
#define INPUT1_TYPE int8_t

#define OUTPUTS_NUM 1

#define OUTPUT0_NDIM 4
#define OUTPUT0_DIM0 1
#define OUTPUT0_DIM1 5
#define OUTPUT0_DIM2 5
#define OUTPUT0_DIM3 5
#define OUTPUT0_SIZE 125
#define OUTPUT0_TYPE int32_t

int8_t input0[INPUT0_SIZE] = {
    -13, 39,  112,  111,  -107, 125,  113, 120, -79,  -38,  127, 103,  -19, 74,
    23,  109, 59,   89,   -69,  -107, 124, -82, 8,    -8,   -72, 125,  -83, 72,
    -38, -5,  -107, -113, 49,   24,   -54, 94,  -116, 18,   94,  -126, -95, 7,
    39,  41,  98,   51,   -49,  -4,   -92, 124, -106, 95,   0,   -11,  9,   40,
    103, -90, -78,  78,   48,   -118, 106, -5,  -23,  107,  -91, -30,  0,   63,
    -41, -4,  100,  123,  -6,   112,  -44, 88,  26,   -34,  83,  -19,  54,  94,
    126, 27,  90,   24,   -81,  -117, 95,  109, -118, -33,  -40, 65,   70,  96,
    105, -68, -14,  -78,  2,    -21,  -96, 67,  26,   -103, 69,  -16,  29,  85,
    98,  50,  -125, 8,    9,    2,    -73, 122, 113,  78,   -83, -52,  -49};

int8_t input1[INPUT1_SIZE] = {
    111,  56,   114, 74,   86,  51,   -12,  -63, -50, -10,  65,   -8,  64,
    64,   -7,   -44, -51,  -72, 70,   9,    35,  -40, 33,   69,   -27, 94,
    30,   36,   98,  -80,  -95, -16,  -79,  -17, 102, 107,  -18,  -90, 71,
    -127, -107, -12, -106, -66, -23,  -106, 93,  95,  127,  30,   -30, 84,
    31,   -128, 47,  41,   94,  60,   83,   56,  -40, -119, -9,   93,  70,
    37,   109,  63,  55,   3,   83,   94,   91,  -64, 51,   -120, 97,  -49,
    105,  51,   86,  36,   102, -117, -53,  89,  -17, 50,   -71,  123, -114,
    15,   -26,  92,  -15,  -80, -56,  83,   14,  -95, 124,  -44,  -38, 69,
    110,  94,   -70, -9,   120, 51,   38,   74,  63,  -48,  -12,  53,  -107,
    50,   -73,  48,  18,   112, 42,   62,   14};

void *inputs[INPUTS_NUM] = {input0, input1};
uint32_t inputs_size[INPUTS_NUM] = {INPUT0_SIZE, INPUT1_SIZE};
uint32_t inputs_elem_size[INPUTS_NUM] = {sizeof(INPUT0_TYPE),
                                         sizeof(INPUT1_TYPE)};

int32_t output0[OUTPUT0_SIZE] = {
    98,   95,   226, 185,  -21,  176,  101, 57,  -129, -48,  192,  95,   45,
    138,  16,   65,  8,    17,   1,    -98, 159, -122, 41,   61,   -99,  219,
    -53,  108,  60,  -85,  -202, -129, -30, 7,   48,   201,  -134, -72,  165,
    -253, -202, -5,  -67,  -25,  75,   -55, 44,  91,   35,   154,  -136, 179,
    31,   -139, 56,  81,   197,  -30,  5,   134, 8,    -237, 97,   88,   47,
    144,  18,   33,  55,   66,   42,   90,  191, 59,   45,   -8,   53,   39,
    131,  17,   169, 17,   156,  -23,  73,  116, 73,   74,   -152, 6,    -19,
    124,  -144, 59,  -55,  -15,  14,   179, 119, -163, 110,  -122, -36,  48,
    14,   161,  -44, -112, 189,  35,   67,  159, 161,  2,    -137, 61,   -98,
    52,   -146, 170, 131,  190,  -41,  10,  -35};

void *outputs[OUTPUTS_NUM] = {output0};
uint32_t outputs_size[OUTPUTS_NUM] = {OUTPUT0_SIZE};
uint32_t outputs_elem_size[OUTPUTS_NUM] = {sizeof(OUTPUT0_TYPE)};

#endif // _TEST_INCLUDE_GUARD_
