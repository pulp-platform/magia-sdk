// ============================================================================
// MAGIA NoC Pressure Benchmark
// ----------------------------------------------------------------------------
// This benchmark measures Network-on-Chip pressure in two scenarios:
//
// 1) Row Pressure:
//    Only leftmost tiles (x_id == 0) fetch from L2 and broadcast
//    to tiles in the same row.
//
// 2) Full Mesh Pressure:
//    All tiles fetch simultaneously from L2 and exchange data.
//
// It measures cycles and verifies correctness of send/receive.
//
// No floating point used (avoids linker errors).
// Fully synchronized (avoids IDMA busy errors).
// Safe L1/L2 addressing.
//
// ============================================================================

#include <stdint.h>

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"

#define WAIT_MODE     WFE

#define BUF_SIZE      256

/* Safe L1 offsets */
#define L1_SRC_OFFSET 0x00020000
#define L1_DST_OFFSET 0x00021000

/* Valid L2 base */
#define L2_BASE       0xCC040000

volatile uint32_t row_cycles[MESH_X_TILES * MESH_Y_TILES];
volatile uint32_t full_cycles[MESH_X_TILES * MESH_Y_TILES];

int main(void)
{
    uint32_t hartid = get_hartid();
    uint32_t x_id   = GET_X_ID(hartid);
    uint32_t y_id   = GET_Y_ID(hartid);

    uint32_t l1_base = get_l1_base(hartid);

    volatile uint8_t *l1_src = (uint8_t *)(l1_base + L1_SRC_OFFSET);
    volatile uint8_t *l1_dst = (uint8_t *)(l1_base + L1_DST_OFFSET);

    volatile uint8_t *l2_row_base = (uint8_t *)(L2_BASE + y_id * 0x10000);

    volatile uint8_t *l2_full_base = (uint8_t *)(L2_BASE + 0x800000 + hartid * 0x10000);

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

    // ---------------- Initialize L1 ----------------

    for (int i = 0; i < BUF_SIZE; i++) {
        l1_src[i] = (uint8_t)(hartid + i);
        l1_dst[i] = 0;
    }

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // =========================================================================
    // ============================ ROW PRESSURE ===============================
    // =========================================================================

    uint32_t row_start = perf_get_cycles();

    if (x_id == 0) {

        // Leftmost tile writes pattern to L2
        for (int i = 0; i < BUF_SIZE; i++)
            l2_row_base[i] = (uint8_t)(y_id + i);

        // Send to row neighbors
        for (int col = 1; col < MESH_X_TILES; col++) {

            uint32_t dest_id = GET_ID(y_id, col);
            uint32_t dest_l1 = get_l1_base(dest_id) + L1_DST_OFFSET;

            idma_memcpy_1d(&idma_ctrl, 1, dest_l1, (uint32_t)l2_row_base, BUF_SIZE);

            eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
        }
    }

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    uint32_t row_stop  = perf_get_cycles();
    row_cycles[hartid] = row_stop - row_start;

    // ---------------- Row Verification ----------------

    uint32_t errors = 0;

    if (x_id != 0) {
        for (int i = 0; i < BUF_SIZE; i++) {
            uint8_t expected = (uint8_t)(y_id + i);
            if (l1_dst[i] != expected)
                errors++;
        }
    }

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // =========================================================================
    // ========================== FULL MESH PRESSURE ===========================
    // =========================================================================

    uint32_t full_start = perf_get_cycles();

    // Each tile writes unique L2 pattern
    for (int i = 0; i < BUF_SIZE; i++)
        l2_full_base[i] = (uint8_t)(hartid + i);

    idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)l2_full_base, (uint32_t)l1_dst, BUF_SIZE);

    eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    uint32_t full_stop  = perf_get_cycles();
    full_cycles[hartid] = full_stop - full_start;

    // ---------------- Full Verification ----------------

    for (int i = 0; i < BUF_SIZE; i++) {
        uint8_t expected = (uint8_t)(hartid + i);
        if (l1_dst[i] != expected)
            errors++;
    }

    // =========================================================================
    // ============================== SUMMARY ==================================
    // =========================================================================

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    printf("Tile %u | Row cycles: %u | Full cycles: %u | Errors: %u\n",
           hartid,
           row_cycles[hartid],
           full_cycles[hartid],
           errors);

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    if (hartid == 0) {

        uint32_t total_row  = 0;
        uint32_t total_full = 0;

        for (int i = 0; i < MESH_X_TILES * MESH_Y_TILES; i++) {
            total_row += row_cycles[i];
            total_full += full_cycles[i];
        }

        printf("\n=== Mesh Summary ===\n");
        printf("Total Row Cycles  : %u\n", total_row);
        printf("Total Full Cycles : %u\n", total_full);
        printf("Full/Row Ratio    : %u\n", (total_row == 0) ? 0 : (total_full / total_row));
    }

    return errors;
}