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
#define INPUT0_DIM1 3
#define INPUT0_DIM2 224
#define INPUT0_DIM3 224
#define INPUT0_SIZE 150528
#define INPUT0_TYPE float16

#define INPUT1_NDIM 4
#define INPUT1_DIM0 1
#define INPUT1_DIM1 3
#define INPUT1_DIM2 224
#define INPUT1_DIM3 224
#define INPUT1_SIZE 150528
#define INPUT1_TYPE float16


#define OUTPUTS_NUM 1

#define OUTPUT0_NDIM 4
#define OUTPUT0_DIM0 1
#define OUTPUT0_DIM1 3
#define OUTPUT0_DIM2 224
#define OUTPUT0_DIM3 224
#define OUTPUT0_SIZE 150528
#define OUTPUT0_TYPE float16


extern float16 input0[INPUT0_SIZE];

extern float16 input1[INPUT1_SIZE];

extern void* inputs[INPUTS_NUM];
extern uint32_t inputs_size[INPUTS_NUM];
extern uint32_t inputs_elem_size[INPUTS_NUM];

extern float16 output0[OUTPUT0_SIZE];

extern void* outputs[OUTPUTS_NUM];
extern uint32_t outputs_size[OUTPUTS_NUM];
extern uint32_t outputs_elem_size[OUTPUTS_NUM];

#endif // _TEST_INCLUDE_GUARD_
