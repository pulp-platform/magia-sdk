// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#include <stdint.h>
#include "mg_spatz.h"

static uint32_t mg_spatz_ready __attribute__((section(".tile_bss")));

/* Start of the Spatz binary, set by the linker script. */
extern uint32_t _spatz_binary_start;

void mg_spatz_init(void)
{
    if (mg_spatz_ready)
        return;
    eu_spatz_init(0);
    spatz_init((uint32_t)&_spatz_binary_start);
    mg_spatz_ready = 1;
}
