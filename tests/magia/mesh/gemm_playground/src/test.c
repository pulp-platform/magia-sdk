#include <stdint.h>
#include "test.h"

#include "tile.h"
#include "gemm_utils.h"
#include "fsync.h"
#include "idma.h"
#include "redmule.h"
#include "eventunit.h"
//#include "utils/l1_fifo.h"

#define WAIT_MODE            WFE
#define abs_threshold_millis 8 /* 0.008 expressed as integer millis */

#define GEMM1_N_TILES        1
// #define GEMM2_N_TILES        6
// #define GEMM3_N_TILES        3
// #define GEMM4_N_TILES        6

static const uint32_t gemm1_tiles[GEMM1_N_TILES] = {0};
// static const uint32_t gemm2_tiles[GEMM2_N_TILES] = {4, 5, 8, 9, 12, 13};
// static const uint32_t gemm3_tiles[GEMM3_N_TILES] = {1, 2, 3};
// static const uint32_t gemm4_tiles[GEMM4_N_TILES] = {6, 7, 10, 11, 14, 15};

#ifndef FIFO_N_CHUNKS
#define FIFO_N_CHUNKS 5
#endif
#define FIFO_BATCH_FRAC (1.0f / FIFO_N_CHUNKS)

static int get_local_idx(uint32_t hartid, const uint32_t *tiles, uint32_t n_tiles)
{
    for (uint32_t i = 0; i < n_tiles; i++)
        if (tiles[i] == hartid)
            return (int)i;
    return -1;
}

/* Zero `n_halfwords` fp16 elements starting at L1 address `base` using 32-bit
 * stores. Assumes `base` is 4-byte aligned and `n_halfwords` is even, which
 * holds for the ping-pong chunks below (chunk_rows * DIM_C with even DIM_C). */
static inline void l1_zero_fp16(uint32_t base, uint32_t n_halfwords)
{
    volatile uint32_t *p = (volatile uint32_t *)base;
    uint32_t n           = n_halfwords >> 1;
    for (uint32_t i = 0; i < n; i++)
        p[i] = 0;
}

int main(void)
{
    /* ~~~~~~~~~~~~~~~~~~~~ 0. Initialization ~~~~~~~~~~~~~~~~~~~~ */
    uint32_t hartid       = get_hartid();
    uint32_t l1_tile_base = get_l1_base(hartid);

    /* Init iDMA */
    idma_config_t idma_cfg      = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg  = &idma_cfg,
        .api  = &idma_api,
    };
    idma_init(&idma_ctrl);

    /* Init RedMulE */
    redmule_config_t redmule_cfg      = {.hartid = hartid};
    redmule_controller_t redmule_ctrl = {
        .base = NULL,
        .cfg  = &redmule_cfg,
        .api  = &redmule_api,
    };
    redmule_init(&redmule_ctrl);

    /* Init FractalSync */
    fsync_config_t fsync_cfg      = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };
    fsync_init(&fsync_ctrl);

/* Init Event Unit */
#if STALLING == 0
    eu_config_t eu_cfg      = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg  = &eu_cfg,
        .api  = &eu_api,
    };

    eu_init(&eu_ctrl);
    eu_clear_events(0xFFFFFFFF);
    eu_fsync_init(&eu_ctrl, 0);
    eu_idma_init(&eu_ctrl, 0);
    eu_redmule_init(&eu_ctrl, 0);
#endif

    int gemm1_idx = get_local_idx(hartid, gemm1_tiles, GEMM1_N_TILES);
    // int gemm2_idx = get_local_idx(hartid, gemm2_tiles, GEMM2_N_TILES);
    // int gemm3_idx = get_local_idx(hartid, gemm3_tiles, GEMM3_N_TILES);
    // int gemm4_idx = get_local_idx(hartid, gemm4_tiles, GEMM4_N_TILES);

    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* ------------------------------------------------------------------ */
    /* GEMM1: R1 = M1 @ M2                                                */
    /* ------------------------------------------------------------------ */
    // if (gemm1_idx >= 0) {
    //     // Compute chunks
    //     uint32_t chunk_rows  = DIM_A * FIFO_BATCH_FRAC;
    //     uint32_t chunk_bytes = chunk_rows * DIM_B * 2;

    //     /* Define workspace: [M2 | M1_pp[0] | M1_pp[1] | R1_pp[0] | R1_pp[1]] */
    //     uint32_t obi_m2    = l1_tile_base;
    //     uint32_t obi_m1[2] = {obi_m2 + DIM_B * DIM_C * 2, obi_m2 + DIM_B * DIM_C * 2 +
    //     chunk_bytes}; uint32_t obi_r1[2] = {obi_m1[1] + chunk_bytes,
    //                           obi_m1[1] + chunk_bytes + chunk_rows * DIM_C * 2};

    //     // Load full M2 from L2
    //     idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m2_inp, obi_m2, DIM_B * DIM_C * 2);

    //     /* redmule_gemm accumulates (Y = X*W + Y); zero both R1 ping-pong
    //      * slots once up front, then re-zero each slot inside the loop after
    //      * its previous DMA-out completes. */
    //     l1_zero_fp16(obi_r1[0], chunk_rows * DIM_C);
    //     l1_zero_fp16(obi_r1[1], chunk_rows * DIM_C);

    //     // Wait for M2 to be fully loaded before starting the ping-pong
    //     eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    //     // Prime the pipeline: load M1_pp[0]
    //     idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m1_inp, obi_m1[0], chunk_bytes);

    //     for (uint32_t i = 0; i < FIFO_N_CHUNKS; i++) {
    //         // Re-zero the slot we're about to reuse (slot (i-2)%2 == i%2)
    //         if (i >= 2) {
    //             l1_zero_fp16(obi_r1[i % 2], chunk_rows * DIM_C);
    //         }

    //         // Wait for M1 chunk i to arrive in L1
    //         eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);

    //         // Compute R1 chunk i
    //         redmule_gemm(
    //             &redmule_ctrl, obi_m1[i % 2], obi_m2, obi_r1[i % 2], chunk_rows, DIM_B, DIM_C);

    //         // Load M1 chunk from L2
    //         if (i + 1 < FIFO_N_CHUNKS) {
    //             idma_memcpy_1d(&idma_ctrl,
    //                            0,
    //                            (uint32_t)m1_inp + (i + 1) * chunk_bytes,
    //                            obi_m1[(i + 1) % 2],
    //                            chunk_bytes);
    //         }

    //         // Wait for RedMulE to finish writing R1[i%2] before DMAing it out
    //         eu_redmule_wait(&eu_ctrl, WAIT_MODE);

    //         if (i > 0) {
    //             // Wait for R1 chunk i-1 to be sent back to L2 (except for the first iteration)
    //             eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    //         }

    //         // Store R1 chunk back to L2 output buffer
    //         idma_memcpy_1d(&idma_ctrl,
    //                        1,
    //                        (uint32_t)r1_out + i * chunk_rows * DIM_C * 2,
    //                        obi_r1[i % 2],
    //                        chunk_rows * DIM_C * 2);
    //     }

    //     eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    // }

    /* ~~~~~~~~~~~~~~~~~~~~ Sandbox Testing ~~~~~~~~~~~~~~~~~~~~ */

    /* Barrier: synchronize all tiles after GEMM1 before sandbox transfers */
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    uint32_t m2_size = (uint32_t)(DIM_B * DIM_C * 2);

    // /* Xfer 1: L2 ? tile 4 L1 */
    // if (hartid == 4) {
    //     idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m2_inp, get_l1_base(4), m2_size);
    //     eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    // }
    // fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    // eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // /* Xfer 2: L2 ? tile 12 L1 */
    // if (hartid == 12) {
    //     idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m2_inp, get_l1_base(12), m2_size);
    //     eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    // }
    // fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    // eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // /* Xfer 3: L2 ? tile 7 L1 */
    // if (hartid == 7) {
    //     idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m2_inp, get_l1_base(7), m2_size);
    //     eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    // }
    // fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    // eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // /* Xfers 4-10: tile 12 L1 ? tiles {7,8,9,11,13,14,15} L1 */
    // static const uint32_t scatter_dsts[] = {7, 8, 9, 11, 13, 14, 15};
    // for (uint32_t k = 0; k < 7; k++) {
    //     if (hartid == 12) {
    //         idma_memcpy_1d(&idma_ctrl, 1, get_l1_base(scatter_dsts[k]), get_l1_base(12),
    //         m2_size); eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    //     }
    //     fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    //     eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    // }

    // /* Xfer 11: tile 12 L1 ? tile 15 L1, pulled exclusively by tile 15 */
    // if (hartid == 15) {
    //     idma_memcpy_1d(&idma_ctrl, 0, get_l1_base(12), get_l1_base(15), m2_size);
    //     eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    // }
    // fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    // eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // /* Xfer 12: tile 12 L1 ? tile 15 L1, split in parallel:
    //  *   tile 12 pushes first half (OBI?AXI, src=tile12 L1, dst=tile15 L1)
    //  *   tile 15 pulls second half (AXI?OBI, src=tile12 L1 + half, dst=tile15 L1 + half) */
    // uint32_t half = m2_size / 2;
    // if (hartid == 12) {
    //     idma_memcpy_1d(&idma_ctrl, 1, get_l1_base(15), get_l1_base(12), half);
    //     eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    // }
    // if (hartid == 15) {
    //     idma_memcpy_1d(&idma_ctrl, 0, get_l1_base(12) + half, get_l1_base(15) + half, half);
    //     eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    // }
    // fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    // eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* Xfer A: L2 ? tile 12 L1 */
    if (hartid == 12) {
        idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m2_inp, get_l1_base(12), m2_size);
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    }
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* Xfer B: L2 ? tile 13 L1 */
    // if (hartid == 13) {
    //     idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m2_inp, get_l1_base(13), m2_size);
    //     eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    // }
    // fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    // eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* Xfer C: tile 12 L1 ? tile 13 L1, pushed by tile 12 */
    if (hartid == 12) {
        idma_memcpy_1d(&idma_ctrl, 1, get_l1_base(13), get_l1_base(12), m2_size);
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    }
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* Xfer D: tile 12 L1 ? tile 13 L1, pulled by tile 13 */
    // if (hartid == 13) {
    //     idma_memcpy_1d(&idma_ctrl, 0, get_l1_base(12), get_l1_base(13), m2_size);
    //     eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    // }
    // fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    // eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* Xfer E: tile 12 L1 ? tile 13 L1, work split in half:
     *   tile 12 pushes first half  (dir=1, src=tile12 L1,        dst=tile13 L1)
     *   tile 13 pulls second half  (dir=0, src=tile12 L1 + half, dst=tile13 L1 + half) */
    uint32_t half = m2_size / 2;
    if (hartid == 12) {
        idma_memcpy_1d(&idma_ctrl, 1, get_l1_base(13), get_l1_base(12), half);
        eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    }
    if (hartid == 13) {
        idma_memcpy_1d(&idma_ctrl, 0, get_l1_base(12) + half, get_l1_base(13) + half, half);
        eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    }
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* Xfer F: L2 ? tile 14 L1 */
    // if (hartid == 14) {
    //     idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m2_inp, get_l1_base(14), m2_size);
    //     eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    // }
    // fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    // eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // /* Xfer G: L2 ? tile 15 L1 */
    // if (hartid == 15) {
    //     idma_memcpy_1d(&idma_ctrl, 0, (uint32_t)m2_inp, get_l1_base(15), m2_size);
    //     eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    // }
    // fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    // eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // /* Xfer H: tile 14 L1 ? tile 15 L1, pushed by tile 14 */
    // if (hartid == 14) {
    //     idma_memcpy_1d(&idma_ctrl, 1, get_l1_base(15), get_l1_base(14), m2_size);
    //     eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    // }
    // fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    // eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // /* Xfer I: tile 14 L1 ? tile 15 L1, pulled by tile 15 */
    // if (hartid == 15) {
    //     idma_memcpy_1d(&idma_ctrl, 0, get_l1_base(14), get_l1_base(15), m2_size);
    //     eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    // }
    // fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    // eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // /* Xfer J: tile 14 L1 ? tile 15 L1, work split in half:
    //  *   tile 14 pushes first half  (dir=1, src=tile14 L1,        dst=tile15 L1)
    //  *   tile 15 pulls second half  (dir=0, src=tile14 L1 + half, dst=tile15 L1 + half) */
    // if (hartid == 14) {
    //     idma_memcpy_1d(&idma_ctrl, 1, get_l1_base(15), get_l1_base(14), half);
    //     eu_idma_wait_o2a(&eu_ctrl, WAIT_MODE);
    // }
    // if (hartid == 15) {
    //     idma_memcpy_1d(&idma_ctrl, 0, get_l1_base(14) + half, get_l1_base(15) + half, half);
    //     eu_idma_wait_a2o(&eu_ctrl, WAIT_MODE);
    // }
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* ~~~~~~~~~~~~~~~~~~~~ Validation ~~~~~~~~~~~~~~~~~~~~ */
    // Final barrier
    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    // Tile 0 checks R1 against golden
    uint32_t errors = 0;

    // if (hartid == 0) {
    //     for (uint32_t i = 0; i < DIM_A; i++) {
    //         for (uint32_t j = 0; j < DIM_C; j++) {
    //             float16 computed = *(volatile float16 *)(&r1_out[i * DIM_C + j]);
    //             float16 expected = r1_golden[i * DIM_C + j];

    //             uint16_t uc = *(uint16_t *)&computed;
    //             uint16_t ue = *(uint16_t *)&expected;

    //             int32_t vc = fp16_to_millis(uc);
    //             int32_t ve = fp16_to_millis(ue);

    //             int32_t abs_diff = vc - ve;
    //             if (abs_diff < 0)
    //                 abs_diff = -abs_diff;
    //             if (abs_diff > abs_threshold_millis) {
    //                 errors++;
    //                 // printf("R1[%d][%d]: got=%f (0x%x) exp=%f (0x%x) (abs_diff=%ld)\n",
    //                 //        i,
    //                 //        j,
    //                 //        fp16_to_f64(uc),
    //                 //        uc,
    //                 //        fp16_to_f64(ue),
    //                 //        ue,
    //                 //        (long)abs_diff);
    //             }
    //         }
    //     }

    //     printf("\nTest complete. Errors: %d / %d\n\n", errors, DIM_A * DIM_C);
    // }

    return errors;
}