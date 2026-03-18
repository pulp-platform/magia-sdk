// Copyright 2025 University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>
// Alex Marchioni <alex.marchioni@chips.it>

#include <stdint.h>
#include "network.h"
#include "test.h"
#include "eventunit.h"
#include "idma.h"
#include "redmule.h"
#include "tile.h"
#include "fsync.h"

#define WAIT_MODE WFE


// TODO: replace with the function commented below when memcmp is available
// this only works if the output vectors are int32_t
uint32_t arrays_equal(const int32_t *expected, const int32_t *computed,
                      uint32_t elem_size, uint32_t length) {
    uint32_t errors = 0;
    for (uint32_t i = 0; i < length; i++) {
        if (computed[i] != expected[i]) {
            printf("Expected %d computed: %d (at index %u)\n", expected[i],
                   computed[i], i);
            errors++;
        }
    }
    return errors;
}
// uint32_t arrays_equal(const void *expected, const void *computed,
//                       uint32_t elem_size, uint32_t length) {
//     uint32_t errors = 0;
//     for (uint32_t i = 0; i < length; i++) {
//         const void *_expected = (const char *)expected + i * elem_size;
//         const void *_computed = (const char *)computed + i * elem_size;
//         if (memcmp(_expected, _computed, elem_size) != 0) {
//             printf("Expected %d computed: %d (at index %u)\n", expected,
//                    computed, i);
//             errors++;
//         }
//     return errors;
// }


int main(void) {

    uint32_t hartid = get_hartid();
    uint32_t l1_tile_base = get_l1_base(hartid);
    uint32_t cycle_start, cycle_stop;

    /* Init tile's iDMA, Redmule, fsync, event-unit */
    idma_config_t idma_cfg = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg = &idma_cfg,
        .api = &idma_api,
    };

    redmule_config_t redmule_cfg = {.hartid = hartid};
    redmule_controller_t redmule_ctrl = {
        .base = NULL,
        .cfg = &redmule_cfg,
        .api = &redmule_api,
    };

    fsync_config_t fsync_cfg = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg = &fsync_cfg,
        .api = &fsync_api,
    };

    fsync_init(&fsync_ctrl);
    idma_init(&idma_ctrl);
    redmule_init(&redmule_ctrl);

    eu_config_t eu_cfg = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg = &eu_cfg,
        .api = &eu_api,
    };
    eu_init(&eu_ctrl);
    eu_fsync_init(&eu_ctrl, 0);
    eu_redmule_init(&eu_ctrl, 0);
    eu_idma_init(&eu_ctrl, 0);

    /* initialization */
    if (hartid == 0) {
        InitNetwork();
    }

    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* input copy */
    // TODO: check if memcopy is necessary!!!
    if (hartid == 0) {
        for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
            memcpy(DeeployNetwork_inputs[buf], inputs[buf],
                   DeeployNetwork_inputs_bytes[buf]);
        }
    }

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* execution */
    cycle_start = perf_get_cycles();
    RunNetwork();
    cycle_stop = perf_get_cycles();
    printf("id: %d, cycles: %d\n", hartid, cycle_stop - cycle_start);

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* comparison */
    uint32_t total_errors = 0;
    if (hartid == 0) {
        for (uint32_t i = 0; i < OUTPUTS_NUM; i++) {
            uint32_t errors =
                arrays_equal(outputs[i], DeeployNetwork_outputs[i],
                             outputs_elem_size[i], outputs_size[i]);
            total_errors += errors;
            printf("output_%d -> number of errors: %d\n", i, errors);
        }
    }

    return total_errors;
}