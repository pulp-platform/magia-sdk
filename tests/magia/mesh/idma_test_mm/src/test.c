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
 * Authors: Luca Balboni <luca.balboni10@studio.unibo.it>
 *         Based on idma_test.c by Victor Isachi
 * 
 * MAGIA iDMA Test using Memory-Mapped Control
 */

#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "idma.h"
#include "eventunit.h"
#include "fsync.h"


#define X_BASE get_l1_base(get_hartid())
#define Y_BASE get_l1_base(get_hartid() + 12288)

#define M_SIZE (96)
#define N_SIZE (64)

#define VERBOSE (0)

#define WAIT_CYCLES (10)

#define CONCURRENT

int main(void) {
  /** 
   * 0. Get the mesh-tile's hartid, mesh-tile coordinates and define its L1 base, 
   * also initialize the controllers for the idma and fsync.
   */
  uint32_t hartid = get_hartid();

  idma_config_t idma_cfg = {.hartid = hartid};
  idma_controller_t idma_ctrl = {
      .base = NULL,
      .cfg = &idma_cfg,
      .api = &idma_api,
  };

  idma_init(&idma_ctrl);

  /* Init FractalSync */
    fsync_config_t fsync_cfg      = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };
    fsync_init(&fsync_ctrl);

  uint32_t y_id = GET_Y_ID(hartid);
  uint32_t x_id = GET_X_ID(hartid);
  uint32_t l1_tile_base = get_l1_base(hartid);

  #if STALLING == 0
  eu_config_t eu_cfg = {.hartid = hartid};
  eu_controller_t eu_ctrl = {
      .base = NULL,
      .cfg = &eu_cfg,
      .api = &eu_api,
  };

  eu_init(&eu_ctrl);
  eu_clear_events(0xFFFFFFFF);
  eu_idma_init(&eu_ctrl, 0);
  eu_fsync_init(&eu_ctrl, 0);
  #endif

  fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
  #if STALLING == 0
  eu_fsync_wait(&eu_ctrl, WFE);
  #endif


  uint32_t dst_addr;
  uint32_t src_addr;
  uint32_t len;

  dst_addr = (uint32_t)X_BASE;
  src_addr = (uint32_t)x_inp;
  len      = (uint32_t)(M_SIZE*N_SIZE*2); // 2 Bytes per element

  printf("iDMA moving data from L2 to L1...\n");
  idma_memcpy_1d(&idma_ctrl, 0, src_addr, dst_addr, len);
  #if STALLING == 0
  eu_idma_wait_a2o(&eu_ctrl, WFE);
  #endif

  dst_addr = (uint32_t)w_out;
  src_addr = (uint32_t)X_BASE;
  len      = (uint32_t)(M_SIZE*N_SIZE*2); // 2 Bytes per element
  printf("iDMA moving data from L1 to L2...\n");
  idma_memcpy_1d(&idma_ctrl, 1, dst_addr, src_addr, len);

  dst_addr = (uint32_t)Y_BASE;
  src_addr = (uint32_t)x_inp;
  len      = (uint32_t)(M_SIZE*N_SIZE*2); // 2 Bytes per element
  // Start both transfers concurrently
  printf("iDMA moving concurrently data from L1 to L2 and from L2 to L1...\n");
  idma_memcpy_1d(&idma_ctrl, 0, src_addr, dst_addr, len);

  // Use WFE to wait for both transfers completion
  #if STALLING == 0
  eu_idma_wait_a2o(&eu_ctrl, WFE);
  #endif
  #if STALLING == 0
  eu_idma_wait_o2a(&eu_ctrl, WFE);
  #endif
  

  printf("Verifying results...\n");
  
  unsigned int num_errors = 0;

  uint16_t detected_l2, expected;
  for(int i = 0; i < M_SIZE*N_SIZE; i++){
    detected_l2 = mmio16((uint32_t)(w_out) + 2*i);
    expected = mmio16((uint32_t)(x_inp) + 2*i);
    if((detected_l2 != expected)){
      num_errors++;
      printf("**ERROR**: DETECTED L2[%d](=0x%x) != EXPECTED[%d](=0x%x)\n", i, detected_l2, i, expected);
    }
  }
  printf("Finished test with %d errors\n", num_errors);

  return num_errors;
}