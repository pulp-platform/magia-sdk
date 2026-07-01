// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ONNX_GLOBALAVERAGEPOOL_PARAMS_H_
#define ONNX_GLOBALAVERAGEPOOL_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t addr_input;
    uintptr_t addr_res;
    uintptr_t addr_exp;
    uint32_t len;
} onnx_globalaveragepool_params_t;

#endif /* ONNX_GLOBALAVERAGEPOOL_PARAMS_H_ */
