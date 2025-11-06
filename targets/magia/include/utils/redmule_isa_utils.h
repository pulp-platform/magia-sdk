/*
 * Copyright (C) 2023-2024 ETH Zurich and University of Bologna
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
 * 
 * MAGIA RedMulE ISA Utils
 */

#ifndef REDMULE_ISA_UTILS_H
#define REDMULE_ISA_UTILS_H

#include "addr_map/tile_addr_map.h"
#include "regs/tile_ctrl.h"

#define HWPE_WRITE(value, offset) *(volatile uint32_t*)(REDMULE_BASE + offset) = value
#define HWPE_READ(offset) *(volatile uint32_t*)(REDMULE_BASE + offset)

#define redmule_wait() __asm__ __volatile__("wfi" ::: "memory")

/* mcnfig instruction */
  // asm volatile(
  //      ".word (0x0       << 25) | \     /* Empty */
  //             (0b00110   << 20) | \     /* Rs2 - t1 */
  //             (0b00101   << 15) | \     /* Rs1 - t0 */
  //             (0x00      <<  7) | \     /* Empty */
  //             (0b0001011 <<  0)   \n"); /* OpCode */
inline void redmule_mcnfig(volatile uint16_t k_size, volatile uint16_t m_size, volatile uint16_t n_size){
  uint32_t cfg_reg0 = (k_size << 16) | (m_size << 0);
  uint32_t cfg_reg1 = n_size << 0;
  asm volatile("addi t0, %0, 0" ::"r"(cfg_reg0));
  asm volatile("addi t1, %0, 0" ::"r"(cfg_reg1));
  asm volatile(
       ".word (0x0       << 25) | \
              (0b00110   << 20) | \
              (0b00101   << 15) | \
              (0x00      <<  7) | \
              (0b0001011 <<  0)   \n");
}

/* marith instruction */
  // asm volatile(
  //     ".word (0b00111   << 27) | \     /* Rs3 - t2 */
  //            (0b00      << 25) | \     /* Empty */
  //            (0b00110   << 20) | \     /* Rs2 - t1 */
  //            (0b00101   << 15) | \     /* Rs1 - t0 */
  //            (0b0       << 14) | \     /* Custom format enable/disable */
  //            (0b0       << 13) | \     /* Widening enable/disable */
  //            (0b001     << 10) | \     /* Operation selection */
  //            (0b001     <<  7) | \     /* Data format */
  //            (0b0101011 <<  0)   \n"); /* OpCode */
inline void redmule_marith(volatile uint32_t y_base, volatile uint32_t w_base, volatile uint32_t x_base){
  asm volatile("addi t2, %0, 0" ::"r"(y_base));
  asm volatile("addi t1, %0, 0" ::"r"(w_base));
  asm volatile("addi t0, %0, 0" ::"r"(x_base));
  asm volatile(
       ".word (0b00111   << 27) | \
              (0b00      << 25) | \
              (0b00110   << 20) | \
              (0b00101   << 15) | \
              (0b0       << 14) | \
              (0b0       << 13) | \
              (0b001     << 10) | \
              (0b001     <<  7) | \
              (0b0101011 <<  0)   \n");
}

inline void redmule_mm_mcnfig(volatile uint16_t k_size, volatile uint16_t m_size, volatile uint16_t n_size){
  uint32_t mcfg_reg0 = (k_size << 16) | (m_size << 0);
  uint32_t mcfg_reg1 = n_size << 0;
  HWPE_WRITE(mcfg_reg0, REDMULE_REG_OFFS + REDMULE_MCFG0_PTR);
  HWPE_WRITE(mcfg_reg1, REDMULE_REG_OFFS + REDMULE_MCFG1_PTR);
}

inline void redmule_mm_marith(volatile uint32_t y_base, volatile uint32_t w_base, volatile uint32_t x_base){
  uint32_t arith_reg = (0b001 << 10) | (0b001 << 7);
  HWPE_WRITE(x_base, REDMULE_REG_OFFS + REDMULE_REG_X_PTR);
  HWPE_WRITE(w_base, REDMULE_REG_OFFS + REDMULE_REG_W_PTR);
  HWPE_WRITE(y_base, REDMULE_REG_OFFS + REDMULE_REG_Z_PTR);
  HWPE_WRITE(arith_reg, REDMULE_REG_OFFS + REDMULE_ARITH_PTR);
  HWPE_WRITE(0, REDMULE_TRIGGER); 
}

#endif /*REDMULE_ISA_UTILS_H*/