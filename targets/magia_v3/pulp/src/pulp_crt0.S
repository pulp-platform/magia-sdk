/*
 * Copyright (C) 2025 ETH Zurich and University of Bologna
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
 * PULP cluster CRT0 for embedded binary in CV32 instrram (magia_v3).
 *
 * Boot sequence:
 *   1. CV32 writes _pulp_binary_start to PULP_BINARY register (0x1744)
 *   2. CV32 enables PULP clock via PULP_CLK_EN (0x1740)
 *   3. All PULP harts reset to _pulp_binary_start and execute _start
 *   4. Each hart computes its local ID, sets up its stack slice, clears BSS
 *   5. Each hart calls main(), then exit()
 *   6. exit() writes 1 to PULP_DONE (0x1748) and halts
 *   7. ClusterRegs counts nb_pulp_cores_to_wait writes then signals CV32 EU
 *
 * Stack layout (absolute tile addresses, grows downward):
 *   STACK_TOP = 0x1C000
 *   core 0: [0x1B800, 0x1C000)  (2 KB)
 *   core 1: [0x1B000, 0x1B800)  (2 KB)
 *   ...
 *   core 7: [0x14000, 0x14800)  (2 KB)
 *
 * NOTE: stack and PULP_DONE addresses are loaded with 'li' (not 'la')
 * because the binary is position-independent (ORIGIN=0x0) and 'la' would
 * produce a PC-relative reference that breaks when loaded at instrram offset.
 */

#ifndef PULP_CORE_COUNT
#define PULP_CORE_COUNT  8
#endif

#define PULP_DONE_ADDR   0x0000174C   /* PULP_CTRL_BASE + 0x0C */
#define STACK_TOP        0x0001C000   /* top of PULP stack area */
#define LOG2_SLICE       11           /* log2(2 KB) — each hart gets 2 KB */

.section .text.start, "ax"
.global _start

_start:
    /* Compute local core ID = mhartid % PULP_CORE_COUNT */
    csrr    a0, mhartid
    li      t0, PULP_CORE_COUNT
    remu    a1, a0, t0

    /* Enable mstatus.MIE and SW interrupt (mie[3]) */
    li      t0, 0x1
    csrrs   zero, mstatus, t0
    li      t0, 0x8
    csrrs   zero, mie, t0

    /* Per-hart stack pointer: sp = STACK_TOP - local_id * 2KB */
    li      x2, STACK_TOP
    slli    t0, a1, LOG2_SLICE
    sub     x2, x2, t0

    /* Clear BSS — all harts write 0, benign race */
    la      t0, __bss_start
    la      t1, __bss_end
1:  bgeu    t0, t1, 2f
    sw      zero, 0(t0)
    addi    t0, t0, 4
    j       1b
2:
    /* Jump to main(argc=0, argv=0) */
    addi    a0, zero, 0
    addi    a1, zero, 0
    la      t2, main
    jalr    x1, t2
    mv      s0, a0

    mv      a0, s0
    la      t2, exit
    jalr    x1, t2

.global _init
.global _fini
_init:
_fini:
    ret

exit:
    /* Signal done to ClusterRegs — each hart writes independently */
    li      a1, PULP_DONE_ADDR
    li      a0, 1
    sw      a0, 0(a1)
    wfi
    j       exit     /* stay halted */

.section .vectors, "ax"
.option norvc
    .org 0x80
    jal x0, _start
