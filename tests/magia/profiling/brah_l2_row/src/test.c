#include <stdint.h>

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"

#define WAIT_MODE     WFE
#define BUF_SIZE      64

#define L2_SRC_BASE   0xCC040000
#define L2_ROW_STRIDE 0x1000 // per-row L2 spacing

volatile uint32_t tile_cycles[MESH_X_TILES * MESH_Y_TILES];

int main(void)
{
    uint32_t hartid = get_hartid();
    uint32_t x_id   = GET_X_ID(hartid);
    uint32_t y_id   = GET_Y_ID(hartid);

    uint32_t l1_base = get_l1_base(hartid);

    uint8_t *l1_buf  = (uint8_t *)(l1_base + 0x2000);
    uint8_t *l1_recv = (uint8_t *)(l1_base + 0x3000);

    // ---------------- Init Controllers ----------------
    idma_config_t idma_cfg      = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg  = &idma_cfg,
        .api  = &idma_api,
    };
    idma_init(&idma_ctrl);

    fsync_config_t fsync_cfg      = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };
    fsync_init(&fsync_ctrl);

    eu_config_t eu_cfg      = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg  = &eu_cfg,
        .api  = &eu_api,
    };
    eu_init(&eu_ctrl);
    eu_idma_init(&eu_ctrl, 0);
    eu_fsync_init(&eu_ctrl, 0);

    // ---------------- L2 buffer for this row ----------------
    uint32_t l2_row_base = L2_SRC_BASE + y_id * L2_ROW_STRIDE;

    // ---------------- Initialize L2 (only left tiles) ----------------
    if (x_id == 0) {
        uint8_t *l2_ptr = (uint8_t *)l2_row_base;
        for (int i = 0; i < BUF_SIZE; i++)
            l2_ptr[i] = (uint8_t)(y_id + i);
    }

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    uint32_t start = perf_get_cycles();

    // ============================================================
    // STEP 1: L2 → Left tile L1
    // ============================================================
    if (x_id == 0) {
        idma_memcpy_1d(&idma_ctrl, 0, l2_row_base, (uint32_t)l1_buf, BUF_SIZE);

        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    }

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // ============================================================
    // STEP 2: Left tile → row neighbors
    // ============================================================
    if (x_id == 0) {

        for (int x = 1; x < MESH_X_TILES; x++) {

            uint32_t dst_id   = GET_ID(y_id, x);
            uint32_t dst_base = get_l1_base(dst_id);

            idma_memcpy_1d(&idma_ctrl, 1, dst_base + 0x2000, (uint32_t)l1_buf, BUF_SIZE);

            eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
        }
    }

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // ============================================================
    // STEP 3: Row tiles modify data
    // ============================================================
    if (x_id != 0) {
        for (int i = 0; i < BUF_SIZE; i++)
            l1_buf[i] += x_id;
    }

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // ============================================================
    // STEP 4: Row tiles → left tile
    // ============================================================
    if (x_id != 0) {

        uint32_t left_id   = GET_ID(y_id, 0);
        uint32_t left_base = get_l1_base(left_id);

        idma_memcpy_1d(
            &idma_ctrl, 1, left_base + 0x3000 + (x_id * BUF_SIZE), (uint32_t)l1_buf, BUF_SIZE);

        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    }

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // ============================================================
    // STEP 5: Left tile writes back to L2
    // ============================================================
    if (x_id == 0) {

        idma_memcpy_1d(&idma_ctrl, 1, l2_row_base, (uint32_t)l1_buf, BUF_SIZE);

        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    }

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    uint32_t stop       = perf_get_cycles();
    tile_cycles[hartid] = stop - start;

    // ============================================================
    // Verification (only left tiles)
    // ============================================================
    uint32_t errors = 0;

    if (x_id == 0) {

        uint8_t *l2_ptr = (uint8_t *)l2_row_base;

        for (int i = 0; i < BUF_SIZE; i++) {

            uint8_t expected = (uint8_t)(y_id + i);
            uint8_t received = l2_ptr[i];

            if (expected != received) {
                errors++;
                printf("Row %d ERROR at %d: exp=%u got=%u\n", y_id, i, expected, received);
            }
        }
    }

    printf("Tile %d cycles: %u errors: %u\n", hartid, tile_cycles[hartid], errors);

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    if (hartid == 0) {
        uint32_t total = 0;
        for (int i = 0; i < MESH_X_TILES * MESH_Y_TILES; i++)
            total += tile_cycles[i];

        printf("Total cycles across mesh: %u\n", total);
    }

    return errors;
}