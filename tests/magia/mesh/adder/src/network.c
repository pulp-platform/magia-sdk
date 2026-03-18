// Copyright 2026 Fondazione ChipsIT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alex Marchioni <alex.marchioni@chips.it>

#include "network.h"
#include "eventunit.h"
#include "idma.h"
#include "redmule.h"
#include "tile.h"

int8_t DeeployNetwork_input_0[125];
int8_t DeeployNetwork_input_1[125];
int32_t DeeployNetwork_output_0[125];
void *DeeployNetwork_inputs[2];
void *DeeployNetwork_outputs[1];

void RunNetwork() {

    // Magia Add (Name: Add, Op: Add)
    MAGIA_add(DeeployNetwork_input_0, DeeployNetwork_input_1,
              DeeployNetwork_output_0, 125, 0);
}

void InitNetwork() {

    // DeeployNetwork_input_0 = (int8_t*) magia_l2_malloc(sizeof(int8_t) * 125);

    // DeeployNetwork_input_1 = (int8_t*) magia_l2_malloc(sizeof(int8_t) * 125);

    // DeeployNetwork_output_0 = (int32_t*) magia_l2_malloc(sizeof(int32_t) *
    // 125);

    DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
    DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
    DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
}
