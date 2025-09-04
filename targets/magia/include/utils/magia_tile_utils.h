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
 * MAGIA Tile Utils
 */

#ifndef MAGIA_TILE_UTILS_H
#define MAGIA_TILE_UTILS_H

#include "tinyprintf.h"

#define mmio64(x) (*(volatile uint64_t *)(x))
#define mmio32(x) (*(volatile uint32_t *)(x))
#define mmio16(x) (*(volatile uint16_t *)(x))
#define mmio8(x)  (*(volatile uint8_t  *)(x))

#define addr64(x) (*(uint64_t *)(&x))
#define addr32(x) (*(uint32_t *)(&x))
#define addr16(x) (*(uint16_t *)(&x))
#define addr8(x)  (*(uint8_t  *)(&x))

static void wait_print(unsigned int cycles){
    for(int i = 0; i <= cycles; i++){
        printf("Waiting: [");
        for(int j = 0; j < i; j++)
            printf("+");
        for(int j = 0; j < cycles - i; j++)
            printf("-");
        printf("]\n");
    }
}

inline void irq_en(volatile uint32_t index_oh){
    asm volatile("addi t0, %0, 0\n\t"
                 "csrrs zero, mie, t0"
                 ::"r"(index_oh));
}

inline uint32_t irq_st(){
    uint32_t irq_status;
    asm volatile("csrr %0, mip"
                 :"=r"(irq_status):);
    return irq_status;
}

inline void wait_nop(uint32_t nops){
    for (int i = 0; i < nops; i++) asm volatile("addi x0, x0, 0" ::);
}

inline void sentinel_instr_id(){
    asm volatile("addi x0, x0, 0x404" ::);
}

inline void sentinel_instr_ex(){
    asm volatile("addi x0, x0, 0x505" ::);
}

static inline void sentinel_start(){
    asm volatile("addi x0, x0, 0x5AA" ::);
}

static inline void sentinel_end(){
    asm volatile("addi x0, x0, 0x5FF" ::);
}

inline void ccount_en(){
    asm volatile("csrrci zero, 0x320, 0x1" ::);
}

inline void ccount_dis(){
    asm volatile("csrrsi zero, 0x320, 0x1" ::);
}

static inline uint32_t get_cyclel(){
    uint32_t cyclel;
    asm volatile("csrr %0, cycle"
                 :"=r"(cyclel):);
    return cyclel;
}

inline uint32_t get_cycleh(){
    uint32_t cycleh;
    asm volatile("csrr %0, cycleh"
                 :"=r"(cycleh):);
    return cycleh;
}

static uint32_t get_cycle(){
    uint32_t cyclel = get_cyclel();
    uint32_t cycleh = get_cycleh();
    if (cycleh) return 0;
    return cyclel;
}

inline uint32_t get_timel(){
    uint32_t timel;
    asm volatile("csrr %0, time"
                 :"=r"(timel):);
    return timel;
}

inline uint32_t get_timeh(){
    uint32_t timeh;
    asm volatile("csrr %0, timeh"
                 :"=r"(timeh):);
    return timeh;
}

static uint32_t get_time(){
    uint32_t timel = get_timel();
    uint32_t timeh = get_timeh();
    if (timeh) return 0;
    return timel;
}

#endif /*MAGIA_TILE_UTILS_H*/