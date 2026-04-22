// Copyright 2026 Fondazione ChipsIT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alex Marchioni <alex.marchioni@chips.it>

#ifndef _NETWORK_INCLUDE_GUARD_
#define _NETWORK_INCLUDE_GUARD_

#include <stdint.h>

void RunNetwork();
void InitNetwork();

extern int8_t DeeployNetwork_input_0[125];
static const uint32_t DeeployNetwork_input_0_len = 125;
extern int8_t DeeployNetwork_input_1[125];
static const uint32_t DeeployNetwork_input_1_len = 125;
extern int32_t DeeployNetwork_output_0[125];
static const uint32_t DeeployNetwork_output_0_len = 125;
static const uint32_t DeeployNetwork_num_inputs = 2;
static const uint32_t DeeployNetwork_num_outputs = 1;
extern void *DeeployNetwork_inputs[2];
extern void *DeeployNetwork_outputs[1];
static const uint32_t DeeployNetwork_inputs_bytes[2] = {125, 125};
static const uint32_t DeeployNetwork_outputs_bytes[1] = {500};

#endif // _NETWORK_INCLUDE_GUARD_
