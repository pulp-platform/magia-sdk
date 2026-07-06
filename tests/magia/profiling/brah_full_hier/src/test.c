// ============================================================================
// MAGIA v2 NoC & L2 Contention Benchmark (Stable / Crash-Free)
//
// Measures:
//   Case A: Private L2 (No Contention)
//   Case B: Row Contention
//   Case C: Full Contention
//
// Round-trip verification:
//   L1 → L2 → L1
//
// Safe addressing:
//   - L2 stays within first 512KB window
//   - No overlap between tiles
//   - No out-of-range access
//
// Fully synchronized.
// No floating point.
// ============================================================================

#include <stdint.h>

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"

#define WAIT_MODE      WFE

#define BUF_SIZE       512

/* SAFE L1 OFFSETS */
#define L1_SRC_OFFSET  0x00020000
#define L1_DST_OFFSET  0x00021000

/* SAFE L2 WINDOW (first 512KB only) */
#define L2_BASE        0xCC040000

/* Keep total usage under 512KB */
#define PRIVATE_STRIDE 0x1000 // 4KB per tile
#define ROW_STRIDE     0x2000 // 8KB per row
#define TILE_OFFSET    0x400  // 1KB inside region

volatile uint32_t private_cycles[MESH_X_TILES * MESH_Y_TILES];
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

    /* SAFE L2 MAPPING */

    volatile uint8_t *l2_private = (uint8_t *)(L2_BASE + hartid * PRIVATE_STRIDE);

    volatile uint8_t *l2_row = (uint8_t *)(L2_BASE + 0x20000 + // move to next 128KB region
                                           y_id * ROW_STRIDE + x_id * TILE_OFFSET);

    volatile uint8_t *l2_full = (uint8_t *)(L2_BASE + 0x40000); // single shared bank region

    /* ================= INIT CONTROLLERS ================= */

    idma_config_t idma_cfg = {.hartid = hartid};
    idma_controller_t idma = {.base = NULL, .cfg = &idma_cfg, .api = &idma_api};
    idma_init(&idma);

    fsync_config_t fs_cfg = {.hartid = hartid};
    fsync_controller_t fs = {.base = NULL, .cfg = &fs_cfg, .api = &fsync_api};
    fsync_init(&fs);

    eu_config_t eu_cfg = {.hartid = hartid};
    eu_controller_t eu = {.base = NULL, .cfg = &eu_cfg, .api = &eu_api};
    eu_init(&eu);
    eu_idma_init(&eu, 0);
    eu_fsync_init(&eu, 0);

    /* ================= INIT L1 ================= */

    for (int i = 0; i < BUF_SIZE; i++) {
        l1_src[i] = (uint8_t)(hartid + i);
        l1_dst[i] = 0;
    }

    fsync_sync_global(&fs);
    eu_fsync_wait(&eu, WAIT_MODE);

    uint32_t errors = 0;

    // =========================================================================
    // CASE A — PRIVATE L2 (NO CONTENTION)
    // =========================================================================

    uint32_t start = perf_get_cycles();

    idma_memcpy_1d(&idma, 0, (uint32_t)l2_private, (uint32_t)l1_src, BUF_SIZE);

    eu_idma_wait_a2o(&eu, WAIT_MODE);

    idma_memcpy_1d(&idma, 1, (uint32_t)l1_dst, (uint32_t)l2_private, BUF_SIZE);

    eu_idma_wait_o2a(&eu, WAIT_MODE);

    uint32_t stop          = perf_get_cycles();
    private_cycles[hartid] = stop - start;

    for (int i = 0; i < BUF_SIZE; i++) {
        if (l1_dst[i] != l1_src[i])
            errors++;
    }

    fsync_sync_global(&fs);
    eu_fsync_wait(&eu, WAIT_MODE);

    // =========================================================================
    // CASE B — ROW CONTENTION
    // =========================================================================

    start = perf_get_cycles();

    idma_memcpy_1d(&idma, 0, (uint32_t)l2_row, (uint32_t)l1_src, BUF_SIZE);

    eu_idma_wait_a2o(&eu, WAIT_MODE);

    idma_memcpy_1d(&idma, 1, (uint32_t)l1_dst, (uint32_t)l2_row, BUF_SIZE);

    eu_idma_wait_o2a(&eu, WAIT_MODE);

    stop               = perf_get_cycles();
    row_cycles[hartid] = stop - start;

    for (int i = 0; i < BUF_SIZE; i++) {
        if (l1_dst[i] != l1_src[i])
            errors++;
    }

    fsync_sync_global(&fs);
    eu_fsync_wait(&eu, WAIT_MODE);

    // =========================================================================
    // CASE C — FULL CONTENTION
    // =========================================================================

    start = perf_get_cycles();

    idma_memcpy_1d(
        &idma, 0, (uint32_t)(l2_full + hartid * TILE_OFFSET), (uint32_t)l1_src, BUF_SIZE);

    eu_idma_wait_a2o(&eu, WAIT_MODE);

    idma_memcpy_1d(
        &idma, 1, (uint32_t)l1_dst, (uint32_t)(l2_full + hartid * TILE_OFFSET), BUF_SIZE);

    eu_idma_wait_o2a(&eu, WAIT_MODE);

    stop                = perf_get_cycles();
    full_cycles[hartid] = stop - start;

    for (int i = 0; i < BUF_SIZE; i++) {
        if (l1_dst[i] != l1_src[i])
            errors++;
    }

    // =========================================================================
    // GLOBAL SUMMARY (Tile 0 Only)
    // =========================================================================

    fsync_sync_global(&fs);
    eu_fsync_wait(&eu, WAIT_MODE);

    printf("Tile %u | Private:%u | Row:%u | Full:%u | Errors:%u\n",
           hartid,
           private_cycles[hartid],
           row_cycles[hartid],
           full_cycles[hartid],
           errors);

    fsync_sync_global(&fs);
    eu_fsync_wait(&eu, WAIT_MODE);

    if (hartid == 0) {
        uint32_t total_private = 0;
        uint32_t total_row     = 0;
        uint32_t total_full    = 0;

        uint32_t total_tiles = MESH_X_TILES * MESH_Y_TILES;

        for (uint32_t i = 0; i < total_tiles; i++) {
            total_private += private_cycles[i];
            total_row += row_cycles[i];
            total_full += full_cycles[i];
        }

        printf("\n==============================\n");
        printf(" MAGIA v2 Contention Summary\n");
        printf("==============================\n");

        printf("Total Private Cycles : %u\n", total_private);
        printf("Total Row Cycles     : %u\n", total_row);
        printf("Total Full Cycles    : %u\n", total_full);

        if (total_private != 0) {
            printf("Row / Private Ratio  : %u\n", total_row / total_private);

            printf("Full / Private Ratio : %u\n", total_full / total_private);
        }

        printf("==============================\n");
    }

    return errors;
}
