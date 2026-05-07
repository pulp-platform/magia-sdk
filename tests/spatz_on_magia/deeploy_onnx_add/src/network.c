// Copyright 2026 Fondazione ChipsIT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// 
// Alex Marchioni <alex.marchioni@chips.it>

#include "tile.h"
#include "idma.h"
#include "redmule.h"
#include "eventunit.h"
#include "network.h"

float16 DeeployNetwork_input_0[150528];
float16 DeeployNetwork_input_1[150528];
float16 DeeployNetwork_output_0[150528];
void* DeeployNetwork_inputs[2];void* DeeployNetwork_outputs[1];

void RunNetwork() {

{

// Magia Onnx Add (Name: , Op: Add)
MAGIA_onnx_add(DeeployNetwork_input_0, DeeployNetwork_input_1, DeeployNetwork_output_0, 150528);

}

}

void InitNetwork() {

// DeeployNetwork_input_0 = (float16*) magia_l2_malloc(sizeof(float16) * 150528);


// DeeployNetwork_input_1 = (float16*) magia_l2_malloc(sizeof(float16) * 150528);


// DeeployNetwork_output_0 = (float16*) magia_l2_malloc(sizeof(float16) * 150528);

DeeployNetwork_inputs[0] = (void*) DeeployNetwork_input_0;DeeployNetwork_inputs[1] = (void*) DeeployNetwork_input_1;DeeployNetwork_outputs[0] = (void*) DeeployNetwork_output_0;}
