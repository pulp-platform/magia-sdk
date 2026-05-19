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
 * PULP cluster CRT0 — task dispatcher for magia_v3.
 *
 * Boot sequence:
 *   1. CV32 writes _pulp_binary_start to PULP_BINARY and broadcasts clock enable
 *      to ALL PULP cores via PULP_CLK_EN (0x1740, write 1)
 *   2. All PULP harts reset to _pulp_binary_start and execute _start
 *   3. Each hart computes its local ID, sets up its own stack slice
 *   4. Core 0 (local ID 0) clears BSS; others spin-wait on a flag
 *   5. All harts set mtvec (direct), enable MEIE + MIE, write 1 to PULP_READY,
 *      then enter the dispatcher loop (WFI — waiting for a task dispatch IRQ)
 *   6. CV32 calls pulp_run_task(fn, core_mask): sets PULP_NB_CORES_TO_WAIT,
 *      writes PULP_TASKBIN (0x1750), then writes one-hot core_mask to PULP_START
 *   7. ClusterRegs fires a 1-cycle MEI edge pulse to each selected PULP core
 *   8. Each woken hart reads TASKBIN and DATA, ACKs by writing 0 to PULP_START
 *      (ClusterRegs counts ACKs and clears the register when all nb_cores_to_wait
 *      ACKs received, unblocking CV32), calls the task, then writes 1 to PULP_DONE;
 *      ClusterRegs counts DONE writes and fires the DONE IRQ to CV32 event unit
 *   9. All harts mret back to dispatcher_loop and wait for the next task
 *
 * NOTE: CV32 should allow enough time between pulp_init() and the first
 * pulp_run_task() for all PULP cores to complete boot and reach WFI.
 *
 * Stack layout (absolute tile addresses, grows downward):
 *   STACK_TOP = 0x1C000
 *   core 0: [0x1B800, 0x1C000)  (2 KB)
 *   core 1: [0x1B000, 0x1B800)  (2 KB)
 *   ...
 *   core 7: [0x14000, 0x14800)  (2 KB)
 */

#ifndef PULP_CORE_COUNT
#define PULP_CORE_COUNT  8
#endif

#define PULP_DONE_ADDR     0x0000174C   /* PULP_CTRL_BASE + 0x0C */
#define PULP_TASKBIN_ADDR  0x00001750   /* PULP_CTRL_BASE + 0x10 */
#define PULP_DATA_ADDR     0x00001754   /* PULP_CTRL_BASE + 0x14 */
#define PULP_START_ADDR    0x00001758   /* PULP_CTRL_BASE + 0x18 */
#define PULP_READY_ADDR    0x0000175C   /* PULP_CTRL_BASE + 0x1C */

#define STACK_TOP          0x0001C000   /* top of PULP stack area */
#define LOG2_SLICE         11           /* log2(2 KB) per hart */

#define MSTATUS_MIE        0x8          /* mstatus.MIE bit */
#define MIE_MEIE           0x800        /* mie.MEIE — machine external interrupt enable */

.section .text.start, "ax"
.global _start

_start:
    /* Compute local core ID = mhartid % PULP_CORE_COUNT */
    csrr    a0, mhartid
    li      t0, PULP_CORE_COUNT
    remu    a1, a0, t0                  /* a1 = local_id */

    /* Per-hart stack: sp = STACK_TOP - local_id * 2KB */
    li      x2, STACK_TOP
    slli    t0, a1, LOG2_SLICE
    sub     x2, x2, t0

    /* BSS clear — only core 0 (local_id == 0) */
    bnez    a1, wait_bss_done
    la      t0, __bss_start
    la      t1, __bss_end
1:  bgeu    t0, t1, bss_done
    sw      zero, 0(t0)
    addi    t0, t0, 4
    j       1b
bss_done:
    li      t0, 1
    la      t1, _bss_cleared_flag
    sw      t0, 0(t1)
    j       setup_irq

wait_bss_done:
    la      t0, _bss_cleared_flag
1:  lw      t1, 0(t0)
    beqz    t1, 1b

setup_irq:
    /* Set mtvec to trap handler (direct mode) */
    la      t0, _trap_handler
    csrw    mtvec, t0

    /* Enable external interrupts in mie */
    li      t0, MIE_MEIE
    csrs    mie, t0

    /* Enable global interrupts in mstatus */
    li      t0, MSTATUS_MIE
    csrs    mstatus, t0

    /* Signal CV32 that this core is booted and waiting for tasks */
    li      t0, PULP_READY_ADDR
    li      t1, 1
    sw      t1, 0(t0)

dispatcher_loop:
    wfi
    j       dispatcher_loop

/* --------------------------------------------------------------- */
/* Trap handler                                                      */
/* --------------------------------------------------------------- */
_trap_handler:
    csrr    t0, mcause
    bltz    t0, _handle_interrupt       /* mcause[31]=1 → interrupt */

_handle_exception:
    /* Unexpected exception: halt this hart */
1:  wfi
    j       1b

_handle_interrupt:
    /* Only handle external interrupt (cause = 11) */
    andi    t0, t0, 0x7FF
    li      t1, 11
    bne     t0, t1, _trap_ret

    /* Read task function address from PULP_TASKBIN */
    li      t0, PULP_TASKBIN_ADDR
    lw      t1, 0(t0)
    beqz    t1, _trap_ret               /* no task set, ignore */

    /* Read data pointer from PULP_DATA — passed as first argument */
    li      t0, PULP_DATA_ADDR
    lw      a0, 0(t0)

    /* Ack to CV32: clear PULP_START BEFORE calling task — unblocks CV32 poll */
    li      t0, PULP_START_ADDR
    sw      zero, 0(t0)

    /* Call task function(data_ptr) */
    jalr    ra, 0(t1)

    /* Signal completion: write 1 to PULP_DONE — ClusterRegs fires DONE IRQ when all cores done */
    li      t0, PULP_DONE_ADDR
    li      t1, 1
    sw      t1, 0(t0)

_trap_ret:
    mret

/* --------------------------------------------------------------- */
/* BSS-cleared synchronisation flag (in .data so it is in instrram) */
/* --------------------------------------------------------------- */
.section .data
.align 2
_bss_cleared_flag:
    .word 0
