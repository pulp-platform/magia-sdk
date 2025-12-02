// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Alberto Dequino <alberto.dequino@unibo.it>
// Luca Balboni <luca.balboni10@studio.unibo.it>
//
// This file provides the implementations for the Event Unit.

#include <stdint.h>
#include "eventunit32.h"
#include "regs/tile_ctrl.h"
#include "addr_map/tile_addr_map.h"
#include "utils/eu_isa_utils.h"
#include "utils/magia_utils.h"
#include "utils/tinyprintf.h"
#include "utils/performance_utils.h"

/**
 * @brief Initialize Event Unit with default configuration
 */
void eu32_init(eu_controller_t *ctrl) {
    // Clear all pending events
    mmio32(EU_CORE_BUFFER_CLEAR) = 0xFFFFFFFF;
    
    // Reset masks to default (disabled)
    mmio32(EU_CORE_MASK) = 0x00000000;
    mmio32(EU_CORE_IRQ_MASK) = 0x00000000;
}

//=============================================================================
// RedMulE-specific Event Functions
//=============================================================================

/**
 * @brief Initialize Event Unit for RedMulE events
 * @param enable_irq If true, enable IRQ for RedMulE completion
 */
void eu32_redmule_init(eu_controller_t *ctrl, uint32_t enable_irq) {
    // Enable RedMulE events in mask
    eu_enable_events(EU_REDMULE_ALL_MASK);
    
    // Optionally enable IRQ for RedMulE completion
    if (enable_irq) {
        eu_enable_irq(EU_REDMULE_DONE_MASK);
    }
}

/**
 * @brief Wait for RedMulE completion using specified mode
 * @param mode Wait mode (polling, WFE, etc.)
 * @return Non-zero if RedMulE completed, 0 if timeout/error
 */
uint32_t eu32_redmule_wait(eu_controller_t *ctrl, eu_wait_mode_t mode) {
    uint32_t retval = eu_wait_events(EU_REDMULE_DONE_MASK, mode, 1000000);
    #if PROFILE_CMP == 1
    stnl_cmp_f();
    #endif
    return retval; // 1M cycle timeout
}

/**
 * @brief Check if RedMulE is currently busy
 * @return Non-zero if RedMulE is busy
 */
uint32_t eu32_redmule_is_busy(eu_controller_t *ctrl) {
    return eu_check_events(EU_REDMULE_BUSY_MASK);
}

/**
 * @brief Check if RedMulE has completed, non-blocking
 * @return Non-zero if RedMulE completed
 */
uint32_t eu32_redmule_is_done(eu_controller_t *ctrl) {
    return eu_check_events(EU_REDMULE_DONE_MASK);
}

//=============================================================================
// iDMA-specific Event Functions  
//=============================================================================

/**
 * @brief Initialize Event Unit for iDMA events
 * @param enable_irq If true, enable IRQ for iDMA completion
 */
void eu32_idma_init(eu_controller_t *ctrl, uint32_t enable_irq) {
    // Enable iDMA events in mask (both directions)
    eu_enable_events(EU_IDMA_ALL_MASK);
    
    // Optionally enable IRQ for iDMA completion (both directions)
    if (enable_irq) {
        eu_enable_irq(EU_IDMA_ALL_DONE_MASK);
    }
}

/**
 * @brief Wait for any iDMA completion using specified mode
 * @param mode Wait mode (polling, WFE, etc.)
 * @return Non-zero if any iDMA completed, 0 if timeout/error
 */
uint32_t eu32_idma_wait(eu_controller_t *ctrl, eu_wait_mode_t mode) {
    return eu_wait_events(EU_IDMA_ALL_DONE_MASK, mode, 1000000); // 1M cycle timeout
}

/**
 * @brief Wait for specific iDMA direction completion
 * @param direction 0 = L2->L1 (A2O), 1 = L1->L2 (O2A)
 * @param mode Wait mode (polling, WFE, etc.)
 * @return Non-zero if specified direction completed, 0 if timeout/error
 */
uint32_t eu32_idma_wait_direction(eu_controller_t *ctrl, uint32_t direction, eu_wait_mode_t mode) {
    uint32_t wait_mask = direction ? EU_IDMA_O2A_DONE_MASK : EU_IDMA_A2O_DONE_MASK;
    uint32_t retval = eu_wait_events(wait_mask, mode, 1000000);
    #if PROFILE_CMO == 1
    if(direction == 1)
        stnl_cmo_f();
    #endif
    #if PROFILE_CMI == 1
    if(direction == 0)
        stnl_cmi_f();
    #endif
    return retval; // 1M cycle timeout
}

/**
 * @brief Wait for L2->L1 (AXI2OBI) completion specifically
 * @param mode Wait mode (polling, WFE, etc.)
 * @return Non-zero if L2->L1 completed, 0 if timeout/error
 */
uint32_t eu32_idma_wait_a2o(eu_controller_t *ctrl, eu_wait_mode_t mode) {
    uint32_t retval = eu_wait_events(EU_IDMA_A2O_DONE_MASK, mode, 1000000);
    #if PROFILE_CMI == 1
    stnl_cmi_f();
    #endif
    return retval;
}

/**
 * @brief Wait for L1->L2 (OBI2AXI) completion specifically  
 * @param mode Wait mode (polling, WFE, etc.)
 * @return Non-zero if L1->L2 completed, 0 if timeout/error
 */
uint32_t eu32_idma_wait_o2a(eu_controller_t *ctrl, eu_wait_mode_t mode) {
    uint32_t retval = eu_wait_events(EU_IDMA_O2A_DONE_MASK, mode, 1000000);
    #if PROFILE_CMO == 1
    stnl_cmo_f();
    #endif
    return retval;
}

/**
 * @brief Check if any iDMA transfer has completed
 * @return Non-zero if any iDMA completed
 */
uint32_t eu32_idma_is_done(eu_controller_t *ctrl) {
    return eu_check_events(EU_IDMA_ALL_DONE_MASK);
}

/**
 * @brief Check if L2->L1 (AXI2OBI) transfer has completed
 * @return Non-zero if L2->L1 completed
 */
uint32_t eu32_idma_a2o_is_done(eu_controller_t *ctrl) {
    return eu_check_events(EU_IDMA_A2O_DONE_MASK);
}

/**
 * @brief Check if L1->L2 (OBI2AXI) transfer has completed
 * @return Non-zero if L1->L2 completed
 */
uint32_t eu32_idma_o2a_is_done(eu_controller_t *ctrl) {
    return eu_check_events(EU_IDMA_O2A_DONE_MASK);
}

/**
 * @brief Check if iDMA has error (using cluster events)
 * @return Non-zero if iDMA error occurred
 */
uint32_t eu32_idma_has_error(eu_controller_t *ctrl) {
    uint32_t events = eu_get_events();
    return events & (EU_IDMA_A2O_ERROR_MASK | EU_IDMA_O2A_ERROR_MASK);
}

/**
 * @brief Check if L2->L1 (AXI2OBI) has error
 * @return Non-zero if L2->L1 error occurred
 */
uint32_t eu32_idma_a2o_has_error(eu_controller_t *ctrl) {
    return eu_check_events(EU_IDMA_A2O_ERROR_MASK);
}

/**
 * @brief Check if L1->L2 (OBI2AXI) has error
 * @return Non-zero if L1->L2 error occurred
 */
uint32_t eu32_idma_o2a_has_error(eu_controller_t *ctrl) {
    return eu_check_events(EU_IDMA_O2A_ERROR_MASK);
}

/**
 * @brief Check if any iDMA transfer is busy
 * @return Non-zero if any iDMA busy
 */
uint32_t eu32_idma_is_busy(eu_controller_t *ctrl) {
    uint32_t events = eu_get_events();
    return events & (EU_IDMA_A2O_BUSY_MASK | EU_IDMA_O2A_BUSY_MASK);
}

/**
 * @brief Check if L2->L1 (AXI2OBI) transfer is busy
 * @return Non-zero if L2->L1 busy
 */
uint32_t eu32_idma_a2o_is_busy(eu_controller_t *ctrl) {
    return eu_check_events(EU_IDMA_A2O_BUSY_MASK);
}

/**
 * @brief Check if L1->L2 (OBI2AXI) transfer is busy
 * @return Non-zero if L1->L2 busy
 */
uint32_t eu32_idma_o2a_is_busy(eu_controller_t *ctrl) {
    return eu_check_events(EU_IDMA_O2A_BUSY_MASK);
}

//=============================================================================
// FSync-specific Event Functions
//=============================================================================

/**
 * @brief Initialize Event Unit for FSync events
 * @param enable_irq If true, enable IRQ for FSync completion
 */
void eu32_fsync_init(eu_controller_t *ctrl, uint32_t enable_irq) {
    // Enable FSync events in mask (bits 25:24)
    eu_enable_events(EU_FSYNC_ALL_MASK);
    
    // Optionally enable IRQ for FSync completion (bit 24)
    if (enable_irq) {
        eu_enable_irq(EU_FSYNC_DONE_MASK);
    }
}

/**
 * @brief Wait for FSync completion using specified mode
 * @param mode Wait mode (polling, WFE, etc.)
 * @return Non-zero if FSync completed, 0 if timeout/error
 */
uint32_t eu32_fsync_wait(eu_controller_t *ctrl, eu_wait_mode_t mode) {
    uint32_t retval = eu_wait_events(EU_FSYNC_DONE_MASK, mode, 1000000);
    #if PROFILE_SNC == 1
    stnl_snc_f();
    #endif
    return retval; // 1M cycle timeout
}

/**
 * @brief Check if FSync has completed
 * @return Non-zero if FSync completed
 */
uint32_t eu32_fsync_is_done(eu_controller_t *ctrl) {
    return eu_check_events(EU_FSYNC_DONE_MASK);
}

/**
 * @brief Check if FSync has error
 * @return Non-zero if FSync error occurred
 */
uint32_t eu32_fsync_has_error(eu_controller_t *ctrl) {
    return eu_check_events(EU_FSYNC_ERROR_MASK);
}

extern void eu_init(eu_controller_t *ctrl)
    __attribute__((alias("eu32_init"), used, visibility("default")));
extern void eu_redmule_init(eu_controller_t *ctrl, uint32_t enable_irq)
    __attribute__((alias("eu32_redmule_init"), used, visibility("default")));
extern uint32_t eu_redmule_wait(eu_controller_t *ctrl, eu_wait_mode_t mode)
    __attribute__((alias("eu32_redmule_wait"), used, visibility("default")));    
extern uint32_t eu_redmule_is_busy(eu_controller_t *ctrl)
    __attribute__((alias("eu32_redmule_is_busy"), used, visibility("default")));
extern uint32_t eu_redmule_is_done(eu_controller_t *ctrl)
    __attribute__((alias("eu32_redmule_is_done"), used, visibility("default")));
extern void eu_idma_init(eu_controller_t *ctrl, uint32_t enable_irq)
    __attribute__((alias("eu32_idma_init"), used, visibility("default")));
extern uint32_t eu_idma_wait_direction(eu_controller_t *ctrl, uint32_t direction, eu_wait_mode_t mode)
    __attribute__((alias("eu32_idma_wait_direction"), used, visibility("default")));
extern uint32_t eu_idma_wait_a2o(eu_controller_t *ctrl, eu_wait_mode_t mode)
    __attribute__((alias("eu32_idma_wait_a2o"), used, visibility("default")));
extern uint32_t eu_idma_wait_o2a(eu_controller_t *ctrl, eu_wait_mode_t mode)
    __attribute__((alias("eu32_idma_wait_o2a"), used, visibility("default")));
extern uint32_t eu_idma_is_done(eu_controller_t *ctrl)
    __attribute__((alias("eu32_idma_is_done"), used, visibility("default")));
extern uint32_t eu_idma_a2o_is_done(eu_controller_t *ctrl)
    __attribute__((alias("eu32_idma_a2o_is_done"), used, visibility("default")));
extern uint32_t eu_idma_o2a_is_done(eu_controller_t *ctrl)
    __attribute__((alias("eu32_idma_o2a_is_done"), used, visibility("default")));
extern uint32_t eu_idma_has_error(eu_controller_t *ctrl)
    __attribute__((alias("eu32_idma_has_error"), used, visibility("default")));
extern uint32_t eu_idma_a2o_has_error(eu_controller_t *ctrl)
    __attribute__((alias("eu32_idma_a2o_has_error"), used, visibility("default")));
extern uint32_t eu_idma_o2a_has_error(eu_controller_t *ctrl)
    __attribute__((alias("eu32_idma_o2a_has_error"), used, visibility("default")));
extern uint32_t eu_idma_is_busy(eu_controller_t *ctrl)
    __attribute__((alias("eu32_idma_is_busy"), used, visibility("default")));
extern uint32_t eu_idma_a2o_is_busy(eu_controller_t *ctrl)
    __attribute__((alias("eu32_idma_a2o_is_busy"), used, visibility("default")));
extern uint32_t eu_idma_o2a_is_busy(eu_controller_t *ctrl)
    __attribute__((alias("eu32_idma_o2a_is_busy"), used, visibility("default")));
extern void eu_fsync_init(eu_controller_t *ctrl, uint32_t enable_irq)
    __attribute__((alias("eu32_fsync_init"), used, visibility("default")));
extern uint32_t eu_fsync_wait(eu_controller_t *ctrl, eu_wait_mode_t mode)
    __attribute__((alias("eu32_fsync_wait"), used, visibility("default")));
extern uint32_t eu_fsync_is_done(eu_controller_t *ctrl)
    __attribute__((alias("eu32_fsync_is_done"), used, visibility("default")));
extern uint32_t eu_fsync_has_error(eu_controller_t *ctrl)
    __attribute__((alias("eu32_fsync_has_error"), used, visibility("default")));

eu_controller_api_t eu_api = {
    .init                   = eu32_init,
    .redmule_init           = eu32_redmule_init,
    .redmule_wait           = eu32_redmule_wait,
    .redmule_is_busy        = eu32_redmule_is_busy,
    .redmule_is_done        = eu32_redmule_is_done,
    .idma_init              = eu32_idma_init,
    .idma_wait_direction    = eu32_idma_wait_direction,
    .idma_wait_a2o          = eu32_idma_wait_a2o,
    .idma_wait_o2a          = eu32_idma_wait_o2a,
    .idma_is_done           = eu32_idma_is_done,
    .idma_a2o_is_done       = eu32_idma_a2o_is_done,
    .idma_o2a_is_done       = eu32_idma_o2a_is_done,
    .idma_has_error         = eu32_idma_has_error,
    .idma_a2o_has_error     = eu32_idma_a2o_has_error,
    .idma_o2a_has_error     = eu32_idma_o2a_has_error,
    .idma_is_busy           = eu32_idma_is_busy,
    .idma_a2o_is_busy       = eu32_idma_a2o_is_busy,
    .idma_o2a_is_busy       = eu32_idma_o2a_is_busy,
    .fsync_init             = eu32_fsync_init,
    .fsync_wait             = eu32_fsync_wait,
    .fsync_is_done          = eu32_fsync_is_done,
    .fsync_has_error        = eu32_fsync_has_error,
};