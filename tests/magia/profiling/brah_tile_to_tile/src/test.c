// Mesh NoC Tile-to-Tile Latency + Verification
// Prints source → destination tile
// sends from right to left

#include <stdint.h>

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"
#include "test.h"

#define WAIT_MODE WFE
#define BUF_SIZE  64

volatile uint32_t tile_cycles[MESH_X_TILES * MESH_Y_TILES];

int main(void)
{
    uint32_t hartid = get_hartid();
    uint32_t x_id   = GET_X_ID(hartid);
    uint32_t y_id   = GET_Y_ID(hartid);

    uint32_t l1_base = get_l1_base(hartid);

    /* ---------------- Controllers ---------------- */

    idma_config_t idma_cfg = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg  = &idma_cfg,
        .api  = &idma_api,
    };
    idma_init(&idma_ctrl);

    fsync_config_t fsync_cfg = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };
    fsync_init(&fsync_ctrl);

    eu_config_t eu_cfg = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg  = &eu_cfg,
        .api  = &eu_api,
    };
    eu_init(&eu_ctrl);
    eu_idma_init(&eu_ctrl, 0);
    eu_fsync_init(&eu_ctrl, 0);

    /* ---------------- Buffers ---------------- */

    uint8_t *l1_src = (uint8_t *)l1_base;
    uint8_t *l1_dst = (uint8_t *)(l1_base + BUF_SIZE);

    for (int i = 0; i < BUF_SIZE; i++)
    {
        l1_src[i] = (uint8_t)(hartid + i);
        l1_dst[i] = 0;
    }

    /* ---------------- Global Sync ---------------- */

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* ---------------- Choose Neighbor ---------------- */
    /* Right neighbor with wrap-around */

    uint32_t next_x  = (x_id + 1) % MESH_X_TILES;
    uint32_t dest_id = GET_ID(y_id, next_x);
    uint32_t remote_l1 = get_l1_base(dest_id);

    printf("Tile %u (%u,%u)  --->  Tile %u (%u,%u)\n",
           hartid, x_id, y_id,
           dest_id, next_x, y_id);

    /* ---------------- Measure DMA ---------------- */

    uint32_t cycle_start = perf_get_cycles();

    idma_memcpy_1d(&idma_ctrl,
                   0,                        // AXI → OBI
                   (uint32_t)l1_src,          // local src
                   remote_l1 + BUF_SIZE,      // remote dst
                   BUF_SIZE);

    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    uint32_t cycle_stop = perf_get_cycles();

    uint32_t my_cycles = cycle_stop - cycle_start;
    tile_cycles[hartid] = my_cycles;

    /* ---------------- Sync before verify ---------------- */

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* ---------------- Verification ---------------- */

    uint32_t errors = 0;

    uint32_t prev_x = (x_id - 1 + MESH_X_TILES) % MESH_X_TILES;
    uint32_t sender_id = GET_ID(y_id, prev_x);

    for (int i = 0; i < BUF_SIZE; i++)
    {
        uint8_t expected = (uint8_t)(sender_id + i);
        uint8_t received = l1_dst[i];

        if (received != expected)
            errors++;
    }

    printf("Tile %u received from Tile %u | cycles: %u | errors: %u\n",
           hartid, sender_id, my_cycles, errors);

    /* ---------------- Global Summary ---------------- */

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    if (hartid == 0)
    {
        uint32_t total_cycles = 0;

        for (int i = 0; i < MESH_X_TILES * MESH_Y_TILES; i++)
            total_cycles += tile_cycles[i];

        printf("Total cycles across mesh: %u\n", total_cycles);
    }

    return errors;
}