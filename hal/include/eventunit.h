// Copyright 2024-2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Alberto Dequino <alberto.dequino@unibo.it>

#ifndef HAL_EU_H
#define HAL_EU_H

typedef enum {
    POLLING = 0, // Busy wait polling
    WFE,         // Wait For Event (RISC-V)
} eu_wait_mode_t;

/**
 * @brief Initialize Event Unit with default configuration
 */
extern void eu_init();

//=============================================================================
// PULP-cluster-specific Event Functions
//=============================================================================

/**
 * @brief Initialize Event Unit for Pulp cluster events
 * @param enable_irq If true, enable IRQ for Pulp-Cluster completion
 */
extern void eu_pulp_init(uint32_t enable_irq);

/**
 * @brief Wait for Pulp cluster completion specifically
 * @param mode Wait mode (polling, WFE, etc.)
 * @return Non-zero if Pulp cluster completed, 0 if timeout/error
 */
extern uint32_t eu_pulp_wait(eu_wait_mode_t mode);

//=============================================================================
// RedMulE-specific Event Functions
//=============================================================================

/**
 * @brief Initialize Event Unit for RedMulE events
 * @param enable_irq If true, enable IRQ for RedMulE completion
 */
extern void eu_redmule_init(uint32_t enable_irq);

/**
 * @brief Wait for RedMulE completion using specified mode
 * @param mode Wait mode (polling, WFE, etc.)
 * @return Non-zero if RedMulE completed, 0 if timeout/error
 */
extern uint32_t eu_redmule_wait(eu_wait_mode_t mode);

/**
 * @brief Check if RedMulE is currently busy
 * @return Non-zero if RedMulE is busy
 */
extern uint32_t eu_redmule_is_busy();

/**
 * @brief Check if RedMulE has completed, non-blocking
 * @return Non-zero if RedMulE completed
 */
extern uint32_t eu_redmule_is_done();

//=============================================================================
// iDMA-specific Event Functions
//=============================================================================

/**
 * @brief Initialize Event Unit for iDMA events
 * @param enable_irq If true, enable IRQ for iDMA completion
 */
extern void eu_idma_init(uint32_t enable_irq);

/**
 * @brief Wait for any iDMA completion using specified mode
 * @param mode Wait mode (polling, WFE, etc.)
 * @return Non-zero if any iDMA completed, 0 if timeout/error
 */
extern uint32_t eu_idma_wait(eu_wait_mode_t mode);

/**
 * @brief Wait for specific iDMA direction completion
 * @param direction 0 = L2->L1 (A2O), 1 = L1->L2 (O2A)
 * @param mode Wait mode (polling, WFE, etc.)
 * @return Non-zero if specified direction completed, 0 if timeout/error
 */
extern uint32_t eu_idma_wait_direction(uint32_t direction, eu_wait_mode_t mode);

/**
 * @brief Wait for L2->L1 (AXI2OBI) completion specifically
 * @param mode Wait mode (polling, WFE, etc.)
 * @return Non-zero if L2->L1 completed, 0 if timeout/error
 */
extern uint32_t eu_idma_wait_a2o(eu_wait_mode_t mode);

/**
 * @brief Wait for L1->L2 (OBI2AXI) completion specifically
 * @param mode Wait mode (polling, WFE, etc.)
 * @return Non-zero if L1->L2 completed, 0 if timeout/error
 */
extern uint32_t eu_idma_wait_o2a(eu_wait_mode_t mode);

/**
 * @brief Check if any iDMA transfer has completed
 * @return Non-zero if any iDMA completed
 */
extern uint32_t eu_idma_is_done();

/**
 * @brief Check if L2->L1 (AXI2OBI) transfer has completed
 * @return Non-zero if L2->L1 completed
 */
extern uint32_t eu_idma_a2o_is_done();

/**
 * @brief Check if L1->L2 (OBI2AXI) transfer has completed
 * @return Non-zero if L1->L2 completed
 */
extern uint32_t eu_idma_o2a_is_done();

/**
 * @brief Check if iDMA has error (using cluster events)
 * @return Non-zero if iDMA error occurred
 */
extern uint32_t eu_idma_has_error();

/**
 * @brief Check if L2->L1 (AXI2OBI) has error
 * @return Non-zero if L2->L1 error occurred
 */
extern uint32_t eu_idma_a2o_has_error();

/**
 * @brief Check if L1->L2 (OBI2AXI) has error
 * @return Non-zero if L1->L2 error occurred
 */
extern uint32_t eu_idma_o2a_has_error();

/**
 * @brief Check if any iDMA transfer is busy
 * @return Non-zero if any iDMA busy
 */
extern uint32_t eu_idma_is_busy();

/**
 * @brief Check if L2->L1 (AXI2OBI) transfer is busy
 * @return Non-zero if L2->L1 busy
 */
extern uint32_t eu_idma_a2o_is_busy();

/**
 * @brief Check if L1->L2 (OBI2AXI) transfer is busy
 * @return Non-zero if L1->L2 busy
 */
extern uint32_t eu_idma_o2a_is_busy();

//=============================================================================
// FSync-specific Event Functions
//=============================================================================

/**
 * @brief Initialize Event Unit for FSync events
 * @param enable_irq If true, enable IRQ for FSync completion
 */
extern void eu_fsync_init(uint32_t enable_irq);

/**
 * @brief Wait for FSync completion using specified mode
 * @param mode Wait mode (polling, WFE, etc.)
 * @return Non-zero if FSync completed, 0 if timeout/error
 */
extern uint32_t eu_fsync_wait(eu_wait_mode_t mode);

/**
 * @brief Check if FSync has completed
 * @return Non-zero if FSync completed
 */
extern uint32_t eu_fsync_is_done();

/**
 * @brief Check if FSync has error
 * @return Non-zero if FSync error occurred
 */
extern uint32_t eu_fsync_has_error();

//=============================================================================
// Spatz-specific Event Functions
//=============================================================================

/**
 * @brief Initialize Event Unit for Spatz events
 */
extern void eu_spatz_init(uint32_t enable_irq);

/**
 * @brief Wait for Spatz completion in WFE mode
 */
extern uint32_t eu_spatz_wait(eu_wait_mode_t mode);

/**
 * @brief Check if Spatz has completed, non-blocking
 * @return Non-zero if Spatz completed
 */
extern uint32_t eu_spatz_is_done();

#endif // HAL_EU_H
