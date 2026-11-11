// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "eventunit.h"
#include "tile.h"

#include "memset_fp16_spatz.h"
#include "memset_fp16_spatz_params.h"
#include "memset_fp16_spatz_task_bin.h"

#define HID         get_hartid()
#define KERNEL_NAME "memset_fp16_spatz"

void MAGIA_memset_zero_fp16_spatz_l1(float16 *C, uint32_t size)
{
    volatile memset_fp16_spatz_params_t params = {
        .shard_C = (uintptr_t)C,
        .len     = size,
    };
    int ret;

    spatz_run_task_with_params(MEMSET_FP16_SPATZ_TASK, (uint32_t)&params);

    ret = eu_spatz_wait(WFE);
    if (ret == 0) {
        printf("[CV32 (%d)] [%s] Wait on Spatz task completion failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
        return;
    }

    ret = spatz_get_exit_code();
    if (ret != 0) {
        printf("[CV32 (%d)] [%s] Spatz task offloading failed with error: %d\n",
               HID,
               KERNEL_NAME,
               ret);
    }
}
