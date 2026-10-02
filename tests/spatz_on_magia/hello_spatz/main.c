// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "tile.h"
#include "eventunit.h"

#include "hello_spatz_task_bin.h"

int main(void)
{
    int errors;

    printf("[CV32] Hello Spatz Test\n");

    errors = 0;

    printf("[CV32] Initializing Event Unit\n");
    eu_init();

    printf("[CV32] Initializing Spatz Event Unit\n");
    eu_spatz_init(0);

    printf("[CV32] Initializing Spatz\n");
    spatz_init(SPATZ_BINARY_START);

    printf("[CV32] Launching SPATZ Task\n");
    spatz_run_task(HELLO_TASK);

    eu_spatz_wait(WFE);

    if (spatz_get_exit_code() != 0) {
        printf("[CV32] SPATZ TASK ENDED with exit code: 0x%03x\n", spatz_get_exit_code());
        errors++;
    } else {
        printf("[CV32] SPATZ TASK ENDED successfully\n");
    }

    spatz_clk_dis();

    return errors;
}
