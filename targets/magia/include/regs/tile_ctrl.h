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
 * Authors: Victor Isachi <victor.isachi@unibo.it>
 * Alberto Dequino <alberto.dequino@unibo.it>
 * 
 * MAGIA Tile Control registers and IRQ
 */

#ifndef _TILE_REG_DEFS_
#define _TILE_REG_DEFS_

#define DEFAULT_EXIT_CODE (0x0)
#define PASS_EXIT_CODE    (0x0)
#define FAIL_EXIT_CODE    (0x1)

#define IRQ_REDMULE_EVT_0 (31)
#define IRQ_REDMULE_EVT_1 (30)
#define IRQ_A2O_ERROR     (29)
#define IRQ_O2A_ERROR     (28)
#define IRQ_A2O_DONE      (27)
#define IRQ_O2A_DONE      (26)
#define IRQ_A2O_START     (25)
#define IRQ_O2A_START     (24)
#define IRQ_A2O_BUSY      (23)
#define IRQ_O2A_BUSY      (22)
#define IRQ_REDMULE_BUSY  (21)
#define IRQ_FSYNC_DONE    (20)
#define IRQ_FSYNC_ERROR   (19)



#endif  // _TILE_REG_DEFS_