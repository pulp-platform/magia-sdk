// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "tile.h"
#include "eventunit.h"

#include "hello_mesh_task_bin.h"

int main(void)
{
    int errors;
    eu_config_t eu_cfg;
    eu_controller_t eu_ctrl;

    printf("[CV32 - HOST] Hello Mesh Test form the HOST!\n");

    errors        = 0;
    eu_cfg.hartid = get_hartid();
    eu_ctrl.base = NULL, eu_ctrl.cfg = &eu_cfg, eu_ctrl.api = &eu_api,

    printf("[CV32 - HOST] Initializing Event Unit\n");
    eu_init(&eu_ctrl);

    printf("[CV32 - HOST] Initializing Mesh Event Unit\n");
    eu_mesh_init(&eu_ctrl, 0);

    printf("[CV32 - HOST] Initializing Mesh\n");
    mesh_init(MESH_BINARY_START);

    printf("[CV32 - HOST] Launching Mesh Task\n");
    mesh_run_task(HELLO_TASK);

    // TO start off, maybe I'll see if I can just use core 0 as host, waiting for the rest of them.
    eu_mesh_wait(&eu_ctrl, WFE);

    if (mesh_get_exit_code() != 0) {
        printf("[CV32 - HOST] MESH TASK ENDED with exit code: 0x%03x\n", spatz_get_exit_code());
        errors++;
    } else {
        printf("[CV32 - HOST] MESH TASK ENDED successfully\n");
    }

    return errors;
}