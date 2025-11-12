/*
 * Copyright (C) 2024 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Authors: Luca Balboni <luca.balboni10@studio.unibo.it>
 * Alberto Dequino <alberto.dequino@unibo.it>
 *
 * MAGIA Event Unit - Generic utilities for all accelerators
 * Supports RedMulE, FSync, iDMA and custom events
 */

#ifndef EVENT_UNIT_UTILS_H
#define EVENT_UNIT_UTILS_H

#include <stdint.h>
#include "magia_tile_utils.h"

//=============================================================================
// Basic Event Unit Control Functions
//=============================================================================

/**
 * @brief Enable specific event types in Event Unit mask
 * @param event_mask Bitmask of events to enable
 */
static inline void eu_enable_events(uint32_t event_mask) {
    mmio32(EU_CORE_MASK_OR) = event_mask;
}

/**
 * @brief Disable specific event types in Event Unit mask
 * @param event_mask Bitmask of events to disable
 */
static inline void eu_disable_events(uint32_t event_mask) {
    mmio32(EU_CORE_MASK_AND) = event_mask;
}

/**
 * @brief Enable IRQ for specific event types
 * @param irq_mask Bitmask of events that should trigger IRQ
 */
static inline void eu_enable_irq(uint32_t irq_mask) {
    mmio32(EU_CORE_IRQ_MASK_OR) = irq_mask;
}

/**
 * @brief Disable IRQ for specific event types
 * @param irq_mask Bitmask of events that should not trigger IRQ
 */
static inline void eu_disable_irq(uint32_t irq_mask) {
    mmio32(EU_CORE_IRQ_MASK_AND) = irq_mask;
}

/**
 * @brief Clear specific events from the buffer
 * @param event_mask Bitmask of events to clear
 */
static inline void eu_clear_events(uint32_t event_mask) {
    mmio32(EU_CORE_BUFFER_CLEAR) = event_mask;
}

/**
 * @brief Get current event buffer (all events)
 * @return 32-bit event buffer value
 */
static inline uint32_t eu_get_events(void) {
    return mmio32(EU_CORE_BUFFER);
}

/**
 * @brief Get current event buffer with mask applied
 * @return 32-bit masked event buffer value
 */
static inline uint32_t eu_get_events_masked(void) {
    return mmio32(EU_CORE_BUFFER_MASKED);
}

/**
 * @brief Get current event buffer with IRQ mask applied
 * @return 32-bit IRQ-masked event buffer value
 */
static inline uint32_t eu_get_events_irq_masked(void) {
    return mmio32(EU_CORE_BUFFER_IRQ_MASKED);
}

/**
 * @brief Check if specific events are present
 * @param event_mask Bitmask of events to check
 * @return Non-zero if any of the specified events are present
 */
static inline uint32_t eu_check_events(uint32_t event_mask) {
    return mmio32(EU_CORE_BUFFER_MASKED) & event_mask;
}

//=============================================================================
// Wait Functions - Different waiting strategies
//=============================================================================

/**
 * @brief Wait for events using polling mode
 * @param event_mask Bitmask of events to wait for
 * @param timeout_cycles Maximum cycles to wait (0 = infinite)
 * @return Non-zero if events detected, 0 if timeout
 */
static inline uint32_t eu_wait_events_polling(uint32_t event_mask, uint32_t timeout_cycles) {
    uint32_t cycles = 0;
    uint32_t detected_events;
    
    do {
        detected_events = eu_check_events(event_mask);
        if (detected_events) {
            eu_clear_events(event_mask);
            return detected_events;
        }
        
        wait_nop(10);
        cycles += 10;
        
    } while (timeout_cycles == 0 || cycles < timeout_cycles);
    
    return 0; // Timeout
}

//=============================================================================
// LOW-LEVEL HAL (PULP-compatible evt_read32)
//=============================================================================

// evt_read32: blocking read with p.elw instruction
static inline unsigned int evt_read32(unsigned int addr) {
    unsigned int value;
    // Direct p.elw inline assembly for PULP cores (RI5CY, CV32E40P)
    __asm__ __volatile__ (
        "p.elw %0, 0(%1)"
        : "=r" (value)
        : "r" (addr)
        : "memory"
    );
    return value;
}


/**
 * @brief Wait for events using RISC-V WFE instruction
 * @param event_mask Bitmask of events to wait for
 * @return Non-zero if events detected
 */
static inline uint32_t eu_wait_events_wfe(uint32_t event_mask) {
    uint32_t detected_events;

    while(eu_check_events(event_mask) == 0){
        evt_read32(EU_CORE_EVENT_WAIT);
    }

    eu_clear_events(event_mask);
    return 1;
}

/**
 * @brief Generic wait function with selectable mode
 * @param event_mask Bitmask of events to wait for
 * @param mode Wait mode (polling, WFE, etc.)
 * @param timeout_cycles Timeout in cycles (polling mode only, 0 = infinite)
 * @return Non-zero if events detected, 0 if timeout
 */
static inline uint32_t eu_wait_events(uint32_t event_mask, int mode, uint32_t timeout_cycles) {
    switch (mode) {
        case 0:
            if(eu_wait_events_polling(event_mask, timeout_cycles) == 0){
                printf("ERROR: TIMEOUT ON POLLING EVENT!\n");
                return 0;
            }
            return 1;
            
        case 1:
            return eu_wait_events_wfe(event_mask);
            
        default:
            printf("ERROR: Unrecognized wait mode.\n");
            return 0;
    }
}

//=============================================================================
// Clock Status Function
//=============================================================================

/**
 * @brief Check Event Unit clock status
 * @return Non-zero if Event Unit clock is enabled
 */
static inline uint32_t eu_clock_is_enabled(void) {
    return mmio32(EU_CORE_STATUS) & 0x1;
}

//=============================================================================
// Software Event Functions
//=============================================================================

/**
 * @brief Trigger a software event
 * @param sw_event_id Software event ID (0-7 typically)
 */
static inline void eu_trigger_sw_event(uint32_t sw_event_id) {
    if (sw_event_id < 8) {
        mmio32(EU_CORE_TRIGG_SW_EVENT + (sw_event_id * 4)) = 1;
    }
}

/**
 * @brief Trigger software event and wait for response
 * @param sw_event_id Software event ID
 * @return Event buffer value after wake-up
 */
static inline uint32_t eu_trigger_sw_event_wait(uint32_t sw_event_id) {
    if (sw_event_id < 8) {
        return mmio32(EU_CORE_TRIGG_SW_EVENT_WAIT + (sw_event_id * 4));
    }
    return 0;
}

#endif /*EVENT_UNIT_UTILS_H*/