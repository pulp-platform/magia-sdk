// Copyright 2024-2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Alberto Dequino <alberto.dequino@unibo.it>

#include <stdint.h>
#include "eventunit.h"


__attribute__((weak)) eu_controller_api_t eu_api = {
    .init                   = eu_init,
    .redmule_init           = eu_redmule_init,
    .redmule_wait           = eu_redmule_wait,
    .redmule_is_busy        = eu_redmule_is_busy,
    .redmule_is_done        = eu_redmule_is_done,
    .idma_init              = eu_idma_init,
    .idma_wait_direction    = eu_idma_wait_direction,
    .idma_wait_a2o          = eu_idma_wait_a2o,
    .idma_wait_o2a          = eu_idma_wait_o2a,
    .idma_is_done           = eu_idma_is_done,
    .idma_a2o_is_done       = eu_idma_a2o_is_done,
    .idma_o2a_is_done       = eu_idma_o2a_is_done,
    .idma_has_error         = eu_idma_has_error,
    .idma_a2o_has_error     = eu_idma_a2o_has_error,
    .idma_o2a_has_error     = eu_idma_o2a_has_error,
    .idma_is_busy           = eu_idma_is_busy,
    .idma_a2o_is_busy       = eu_idma_a2o_is_busy,
    .idma_o2a_is_busy       = eu_idma_o2a_is_busy,
    .fsync_init             = eu_fsync_init,
    .fsync_wait             = eu_fsync_wait,
    .fsync_is_done          = eu_fsync_is_done,
    .fsync_has_error        = eu_fsync_has_error,
};