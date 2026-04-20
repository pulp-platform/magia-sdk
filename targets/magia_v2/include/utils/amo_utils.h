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
 * Authors: Alberto Dequino <alberto.dequino@unibo.it>
 * 
 * MAGIA Atomic Memory Operations Utils
 */

#ifndef AMO_UTILS_H
#define AMO_UTILS_H

#include "magia_tile_utils.h"

/**
 * Atomically increase the value stored in addr by an immediate value
 */
int amo_add_immediate(uint32_t addr, uint32_t immediate){
    asm volatile("addi t0, %0, 0" ::"r"(addr));
    asm volatile("mv t1, %0" ::"r"(immediate));
    asm volatile("amoadd.w t2, t1, (t0)" ::);
    return 0;
}

/**
 * Atomically increase the value in addr by 1
 */
inline void amo_increment(volatile uint32_t addr){
    asm volatile("addi t0, %0, 0" ::"r"(addr));
    asm volatile("li t1, 1" ::);
    asm volatile("amoadd.w t2, t1, (t0)" ::);
}

#endif //AMO_UTILS_H