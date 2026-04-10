/*
 * Copyright (C) 2018-2019 ETH Zurich and University of Bologna
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
 */

/*
 * Authors: Germain Haugou, ETH (germain.haugou@iis.ee.ethz.ch)
 *          Francesco Conti, ETHZ & UNIBO
 */

#include "addr_map/tile_addr_map.h"

  .section .text
  .global _start
_start:

  # Cluster PEs will also starts here to avoid aligning another entry point
  # Just re-route them to the right entry
  csrr    a0, mhartid
  li      t0, PULP_HARTID_BASE
  bltu    a0, t0, _halt_unused_hart
  sub     a1, a0, t0
  li      t0, TOTAL_PULP_HARTS
  bgeu    a1, t0, _halt_unused_hart
  li      t0, PULP_CORE_COUNT
  remu    a1, a1, t0
  # Enabling CV32E40P mstatus.MIE
  li      t0, 0x1
  csrrs   zero, mstatus, t0
  # Enabling CV32E40P SW interrupt (mie[3])
  li      t0, 0x8
  csrrs   zero, mie, t0

  # clear the bss segment
  la      t0, _bss_start
  la      t1, _bss_end
1:
  sw      zero, 0(t0)
  addi    t0, t0, 4
  bltu    t0, t1, 1b

  /* Stack initialization */
  la      x2, stack
  slli    t0, a1, 11
  sub     x2, x2, t0

.section .text

  // On all other chips we simply pass 0.
  addi  a0, x0, 0
  addi  a1, x0, 0

  // Jump to main program entry point (argc = a0, argv = a1).
  la    t2, main
  jalr  x1, t2
  mv    s0, a0

  /* If program returns from main, call exit routine */
  mv   a0, s0
  la   t2, exit
  jalr  x1, t2

  .global _init
  .global _fini
_init:
_fini:
  # These don't have to do anything since we use init_array/fini_array.
  ret

_halt_unused_hart:
  wfi
  j       _halt_unused_hart

exit:
  li      a1, 0x00001744
  li      a0, 1
  sw      a0, 0(a1)
  wfi

.section .vectors, "ax"
.option norvc;

  .org 0x80
  jal x0, _start
