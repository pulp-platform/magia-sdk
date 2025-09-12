// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Viviane Potocnik <vivianep@iis.ee.ethz.ch>
// Alberto Dequino <alberto.dequino@unibo.it>
//
// This file provides the strong (driver-specific) implementations for the
// IDMA functions using a 32-bit representation (a struct with 'low' and 'high').
// These functions override the weak HAL symbols.
// This is a WIP and might be redundant, as the moment of writing there is only one IDMA configuration tested on MAGIA.

#include <stdint.h>
#include "idma32.h"
#include "regs/tile_ctrl.h"
#include "utils/idma_isa_utils.h"
#include "utils/magia_utils.h"
#include "utils/tinyprintf.h"

int idma32_init(idma_controller_t *ctrl) {
    uint32_t index = (1<<IRQ_A2O_DONE) | (1<<IRQ_O2A_DONE);
    irq_en(index);
    return 0;
}

/* static inline __attribute__((always_inline)) void idma32_wait(){
    asm volatile("wfi" ::: "memory");
} */

/**
 * Start 1-dimensional memory copy
 *
 * @param dir Copy Direction. 0 = AXI to OBI (L2 to L1), !0 = OBI to AXI (L1 to L2).
 * @param axi_addr AXI/L2 memory address of first element.
 * @param obi_addr OBI/L1 memory address of first element.
 * @param len Byte length of memory block to transfer.
 */
int idma32_memcpy_1d(idma_controller_t *ctrl, uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len){
    if(dir){ // OBI to AXI (L1 to L2)
        idma_conf_out();
        idma_set_addr_len_out(axi_addr, obi_addr, len);
        idma_set_std2_rep2_out(0, 0, 1);
        idma_set_std3_rep3_out(0, 0, 1);
        idma_start_out();
        //printf("IDMA_memcpy_1d: Detected IRQ...\n");
    }
    else{ // AXI to OBI (L2 to L1)
        idma_conf_in();
        idma_set_addr_len_in(obi_addr, axi_addr, len);
        idma_set_std2_rep2_in(0, 0, 1);
        idma_set_std3_rep3_in(0, 0, 1);
        idma_start_in();
        //printf("IDMA_memcpy_1d: Detected IRQ...\n");
    }
    return 0;
}

/**
 * Start 2-dimensional memory copy.
 *
 * @param dir Copy Direction. 0 = AXI to OBI (L2 to L1), !0 = OBI to AXI (L1 to L2).
 * @param axi_addr AXI/L2 memory address of first element.
 * @param obi_addr OBI/L1 memory address of first element.
 * @param len Byte length of memory block to transfer for each repetition.
 * @param std Offset to add to the memory address (axi_addr or obi_addr) to calculate the start of the next memory block.
 * @param reps Number of repetitions.
 */
int idma32_memcpy_2d(idma_controller_t *ctrl, uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len, uint32_t std, uint32_t reps){
    //printf("IDMA Transfer! Direction: %d\n", dir);
    
    if(dir){ // OBI to AXI (L1 to L2)
        idma_conf_out();
        idma_set_addr_len_out(axi_addr, obi_addr, len);
        idma_set_std2_rep2_out(std, len, reps);
        idma_set_std3_rep3_out(0, 0, 1);
        idma_start_out();
        //printf("IDMA_memcpy_2d: Detected IRQ...\n");
    }
    else{ // AXI to OBI (L2 to L1)
        idma_conf_in();
        idma_set_addr_len_in(obi_addr, axi_addr, len);
        idma_set_std2_rep2_in(len, std, reps);
        idma_set_std3_rep3_in(0, 0, 1);
        idma_start_in();
        //printf("IDMA_memcpy_2d: Detected IRQ...\n");  
    }
    return 0;
}

extern int idma_init(idma_controller_t *ctrl)
    __attribute__((alias("idma32_init"), used, visibility("default")));
/* extern void idma_wait()
    __attribute__((alias("idma32_wait"), used, visibility("default"))); */
extern int idma_memcpy_1d(idma_controller_t *ctrl, uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len)
    __attribute__((alias("idma32_memcpy_1d"), used, visibility("default")));
extern int idma_memcpy_2d(idma_controller_t *ctrl, uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len, uint32_t std, uint32_t reps)
    __attribute__((alias("idma32_memcpy_2d"), used, visibility("default")));


/* Export the IDMA-specific controller API */
idma_controller_api_t idma_api = {
    .init = idma32_init,
/*     .wait = idma32_wait, */
    .memcpy_1d = idma32_memcpy_1d,
    .memcpy_2d = idma32_memcpy_2d,
};