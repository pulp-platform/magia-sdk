// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "tile.h"

int hello_task(void)
{
    printf("[MESH TILE %d] Hello World from Mesh!\n", get_hartid());
    return 0;
}
