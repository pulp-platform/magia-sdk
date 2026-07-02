// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>

#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"

#define WAIT_MODE WFE

/**
 * This test verifies the native 3D IDMA burst (idma_memcpy_3d). Each mesh-tile loads its
 * static output data-tile from L2 into L1 and writes it back to a separate L2 buffer,
 * describing the transfer as a 3D copy: the tile's rows are split into reps3 planes of
 * reps2 rows each. The result is identical to the 2D loopback (test_idma_2d), so a passing
 * final check confirms the std3/rep3 registers are driven correctly.
 */
int main(void){
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

    fsync_config_t fsync_cfg = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg = &fsync_cfg,
        .api = &fsync_api,
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
    eu_fsync_init(&eu_ctrl, 0);
    eu_idma_init(&eu_ctrl, 0);
    #endif

    /**
     * 1. Calculate the static output data-tile dimensions.
     * If M and K are perfect multiples of the number of mesh-tiles in their respective axis,
     * tile_h and tile_w will be the same for each mesh-tile.
     * Otherwise, these dimensions will be smaller for the mesh-tiles closer to the right and lower
     * border of the mesh.
     */
    uint32_t tile_h_max = ((M_SIZE + MESH_Y_TILES - 1) /  MESH_Y_TILES);
    uint32_t tile_w_max = ((K_SIZE + MESH_X_TILES - 1) /  MESH_X_TILES);
    int32_t tile_h;
    int32_t tile_w;

    if(((tile_h_max * y_id) + tile_h_max) > M_SIZE){
        tile_h = M_SIZE - (tile_h_max * y_id);
    }
    else{
        tile_h = tile_h_max;
    }

    if(((tile_w_max * x_id) + tile_w_max) > N_SIZE){
        tile_w = N_SIZE - (tile_w_max * x_id);
    }
    else{
        tile_w = tile_w_max;
    }

    if(tile_h < 1 || tile_w < 1){
        return 0;
    }
    else{
        //printf("ID:%d, Mesh-Tile-X:%d, Mesh-Tile-Y:%d, Data-Tile w: %d, Data-Tile h: %d", hartid, x_id, y_id, tile_w, tile_h);
    }

    /**
     * 2. Use IDMA to transfer output data-tile into L1 memory with a single 3D burst.
     * The tile's tile_h rows are decomposed into reps3 planes of reps2 rows. Rows on the
     * strided L2 side advance by std2 (one matrix row); planes advance by std3 (reps2 rows).
     * The L1 destination is packed contiguously, so rows/planes map back to the same region
     * a 2D copy would produce.
     */
    uint32_t len = tile_w * 2;
    uint32_t std2 = K_SIZE * 2;
    uint32_t reps3 = ((uint32_t)tile_h % 2 == 0) ? 2 : 1;
    uint32_t reps2 = (uint32_t) tile_h / reps3;
    uint32_t std3 = reps2 * K_SIZE * 2;
    uint32_t obi_addr = (l1_tile_base);
    uint32_t axi_addr_z = (uint32_t) z_out + (y_id * K_SIZE * tile_h_max * 2) + (tile_w_max * x_id * 2);
    uint32_t axi_addr_y = (uint32_t) y_inp + (y_id * K_SIZE * tile_h_max * 2) + (tile_w_max * x_id * 2);

    // printf("Copying data from L2\n");
    idma_memcpy_3d(&idma_ctrl, 0, axi_addr_z, obi_addr, len, std2, reps2, std3, reps3);
    #if STALLING == 0
    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    #endif

    // printf("Copying data to L2\n");
    /**
     * 3. Use IDMA to write the L1 data into the input vector in L2.
     */
    idma_memcpy_3d(&idma_ctrl, 1, axi_addr_y, obi_addr, len, std2, reps2, std3, reps3);
    #if STALLING == 0
    eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    #endif


    /**
     * 4. Wait that all the tiles have finished
     */
    fsync_sync_global(&fsync_ctrl);
    #if STALLING == 0
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    #endif


    /**
     * 5. Check results
     */
    // printf("Check start\n");
    volatile uint32_t errors=0;
    volatile uint16_t computed, expected, diff = 0;
    for(uint8_t i = (y_id * tile_h_max); i < (y_id * tile_h_max + tile_h); i++){
        for(uint8_t j = (x_id * tile_w_max); j < (x_id * tile_w_max + tile_w); j++){
            computed = *(volatile uint16_t*)(y_inp + (i * K_SIZE + j));
            expected = *(volatile uint16_t*)(z_out + (i * K_SIZE + j));
            diff = (computed > expected) ? (computed - expected) : (expected - computed);
            if(diff > 0x0011){
                #if EVAL == 1
                if(y_id == 0)
                    printf("Error detected at coordinates[%d][%d]: Y=%x Z=%x\n", i, j, *(volatile uint16_t*)(y_inp+ (i * K_SIZE + j)), *(volatile uint16_t*)(z_out + (i * K_SIZE + j)));
                #endif
                errors++;
            }
        }
    }
    printf("Number of errors: %d\n", errors);

    return errors;
}
