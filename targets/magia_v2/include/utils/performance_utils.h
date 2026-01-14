/*
 * Copyright (C) 2023-2025 ETH Zurich and University of Bologna
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
 * Authors: Victor Isachi <victor.isachi@unibo.it>
 * Alberto Dequino <alberto.dequino@unibo.it>
 * 
 * MAGIA Utils
 */

#ifndef PERFORMANCE_UTILS_H
#define PERFORMANCE_UTILS_H


/**
 * @brief Starts all performance counters
 */
static inline void perf_start(void) {
    // enable all counters
    asm volatile("csrc 0x320, %0" : : "r"(0xffffffff));
    // arbitrary association of one event to one counter,
    // just the implemented ones will increase
    asm volatile("csrw 0x323, %0" : : "r"(1<<2));
}

/**
 * @brief Stops all performance counters
 */
static inline void perf_stop(void) {
    asm volatile("csrw 0x320, %0" : : "r"(0xffffffff));
}

/**
 * @brief Resets all performance counters to 0 without stopping them
 */
static inline void perf_reset(void) {
    asm volatile("csrw 0xB00, %0" : : "r"(0));
    asm volatile("csrw 0xB02, %0" : : "r"(0));
}

/**
 * @brief Returns the cycles of the performance counter
 */
static inline unsigned int perf_get_cycles(){
    unsigned int value = 0;
    asm volatile ("csrr %0, 0xB00" : "=r" (value));
    return value;
}

/**
 * @brief Returns the n. instructions of the performance counter
 */
static inline unsigned int perf_get_instr(){
    unsigned int value = 0;
    asm volatile ("csrr %0, 0xB02" : "=r" (value));
    return value;
}

static inline void sentinel_start(){
    asm volatile("addi x0, x0, 0x5AA" ::);
}

static inline void sentinel_end(){
    asm volatile("addi x0, x0, 0x5FF" ::);
}

// Start input communication sentinel accumulator
static inline void stnl_cmi_s(){
    asm volatile("addi x0, x0, 0x50B" ::);
}

// Start output communication sentinel accumulator
static inline void stnl_cmo_s(){
    asm volatile("addi x0, x0, 0x51B" ::);
}

// Start computation sentinel accumulator
static inline void stnl_cmp_s(){
    asm volatile("addi x0, x0, 0x52B" ::);
}

// Start synchronization sentinel accumulator
static inline void stnl_snc_s(){
    asm volatile("addi x0, x0, 0x53B" ::);
}

// Start timeslot sentinel accumulator
static inline void stnl_ts_s(){
    asm volatile("addi x0, x0, 0x5FB" ::);
}

// Finish (record) input communication sentinel accumulator
static inline void stnl_cmi_f(){
    asm volatile("addi x0, x0, 0x50C" ::);
}

// Finish (record) output communication sentinel accumulator
static inline void stnl_cmo_f(){
    asm volatile("addi x0, x0, 0x51C" ::);
}

// Finish (record) computation sentinel accumulator
static inline void stnl_cmp_f(){
    asm volatile("addi x0, x0, 0x52C" ::);
}

// Finish (record) synchronization sentinel accumulator
static inline void stnl_snc_f(){
    asm volatile("addi x0, x0, 0x53C" ::);
}

// Finish (record) timeslot sentinel accumulator
static inline void stnl_ts_f(){
    asm volatile("addi x0, x0, 0x5FC" ::);
}

// Report input communication sentinel accumulator
static inline void stnl_cmi_r(){
    asm volatile("addi x0, x0, 0x50D" ::);
}

// Report output communication sentinel accumulator
static inline void stnl_cmo_r(){
    asm volatile("addi x0, x0, 0x51D" ::);
}

// Report computation sentinel accumulator
static inline void stnl_cmp_r(){
    asm volatile("addi x0, x0, 0x52D" ::);
}

// Report synchronization sentinel accumulator
static inline void stnl_snc_r(){
    asm volatile("addi x0, x0, 0x53D" ::);
}

// Report timeslot sentinel accumulator
static inline void stnl_ts_r(){
    asm volatile("addi x0, x0, 0x5FD" ::);
}

// Report global input communication, output communication and computation overheads
static inline void stnl_r(){
    asm volatile("addi x0, x0, 0x5EE" ::);
}

#endif