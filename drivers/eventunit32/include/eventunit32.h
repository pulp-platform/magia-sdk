// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Authors: Viviane Potocnik <vivianep@iis.ee.ethz.ch>
//          Alberto Dequino <alberto.dequino@unibo.it>

#pragma once

#include <stdint.h>
#include "eventunit.h"
#include "regs/tile_ctrl.h"          // EU_*_MASK
#include "utils/eu_isa_utils.h"      // eu_wait_events()
#include "utils/performance_utils.h" // stnl_cmp_f()

/**
 * @brief Wait for RedMulE completion using specified mode
 * @param mode Wait mode (polling, WFE, etc.)
 * @return Non-zero if RedMulE completed, 0 if timeout/error
 */
static inline __attribute__((always_inline)) uint32_t eu32_redmule_wait(eu_controller_t *ctrl,
                                                                        eu_wait_mode_t mode)
{
    uint32_t retval = eu_wait_events(EU_REDMULE_DONE_MASK, mode, 1000000);
#if PROFILE_CMP == 1
    stnl_cmp_f();
#endif
    return retval; // 1M cycle timeout
}

/**
 * @brief Wait for any iDMA completion using specified mode
 * @param mode Wait mode (polling, WFE, etc.)
 * @return Non-zero if any iDMA completed, 0 if timeout/error
 */
static inline __attribute__((always_inline)) uint32_t eu32_idma_wait(eu_controller_t *ctrl,
                                                                     eu_wait_mode_t mode)
{
    return eu_wait_events(EU_IDMA_ALL_DONE_MASK, mode, 1000000); // 1M cycle timeout
}

/**
 * @brief Wait for L2->L1 (AXI2OBI) completion specifically
 * @param mode Wait mode (polling, WFE, etc.)
 * @return Non-zero if L2->L1 completed, 0 if timeout/error
 */
static inline __attribute__((always_inline)) uint32_t eu32_idma_wait_a2o(eu_controller_t *ctrl,
                                                                         eu_wait_mode_t mode)
{
    uint32_t retval = eu_wait_events(EU_IDMA_A2O_DONE_MASK, mode, 1000000);
#if PROFILE_CMI == 1
    stnl_cmi_f();
#endif
    return retval; // 1M cycle timeout
}

/**
 * @brief Wait for L1->L2 (OBI2AXI) completion specifically
 * @param mode Wait mode (polling, WFE, etc.)
 * @return Non-zero if L1->L2 completed, 0 if timeout/error
 */
static inline __attribute__((always_inline)) uint32_t eu32_idma_wait_o2a(eu_controller_t *ctrl,
                                                                         eu_wait_mode_t mode)
{
    uint32_t retval = eu_wait_events(EU_IDMA_O2A_DONE_MASK, mode, 1000000);
#if PROFILE_CMO == 1
    stnl_cmo_f();
#endif
    return retval; // 1M cycle timeout
}