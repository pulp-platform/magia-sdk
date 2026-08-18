// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "printf.h"

int hello_task(void)
{
    printf("[SNITCH] Hello World from Spatz!\n");
    return 0;
}
