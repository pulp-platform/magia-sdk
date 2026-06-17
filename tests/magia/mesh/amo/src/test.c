// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>

#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "fsync.h"
#include "eventunit.h"

#define NUM_ITER     (100)
#define INITIAL_VAL  (0)
#define AMO_TILES    (NUM_HARTS)
#define EXPECTED_VAL (INITIAL_VAL + NUM_ITER*AMO_TILES)

int main(void) {
  uint32_t hartid = get_hartid();

  printf("Starting AMO test...\n");

  for (int i = 0; i < NUM_ITER; i++){
    wait_nop(get_hartid());
    for (int i = 0; i < AMO_TILES; i++){
      asm volatile("addi t0, %0, 0" ::"r"((uint32_t)(SYNC_BASE + ((hartid+i)%NUM_HARTS)*L1_TILE_OFFSET)));
      asm volatile("li t1, 1" ::);
      asm volatile("amoadd.w t2, t1, (t0)" ::);
    }
  }

  printf("Waiting for counter to reach expected value...\n");
  while (mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET) != EXPECTED_VAL){
#if VERBOSE > 10
  printf("Read Synch Value: %0d - Expected: %0d\n", mmio32(SYNC_BASE + hartid*L1_TILE_OFFSET), EXPECTED_VAL);
#endif
  }

  printf("Test PASSED: counter reached\n");

  return 0;
}