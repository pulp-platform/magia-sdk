// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// nn_is_bench -- shared contract between the CV32 control core, the Spatz
// vector unit, the PULP cluster and (later) the VFIO host.
//
// One iteration of the kernel is one inference of a fused NN block:
//
//   Z = X * W + bias                (RedMulE, input-stationary over the mesh)
//   P = hardswish(Z * scale + b)    (Spatz, fp16 vector)
//   P = layernorm_rowwise(P)        (Spatz, fp16 vector)
//   Q = softmax_rowwise_q15(P)      (PULP cluster, 8 cores, int16 packed SIMD)
//
// Geometry per mesh tile, derived from tile_config.h (i.e. from `make build
// tiles=N`), so the same sources run on 1x1, 2x2 and 4x4:
//
//   NN_TH  rows of X/Z owned by a tile           = NN_M / MESH_Y_TILES
//   NN_TW  inner-dimension slice owned by a tile = NN_N / MESH_X_TILES
//   NN_TS  K slice processed per timeslot        = NN_K / NN_TIMESLOTS
//   NN_PB  columns of the post-GEMM block        = NN_K / MESH_X_TILES
//
// The GEMM phase follows the mm_is dataflow: X stays resident in L1 while W is
// streamed timeslot by timeslot and the partial Z travels left-to-right along
// each mesh row, handed over with FractalSync neighbour syncs.

#ifndef NN_IS_PARAMS_H_
#define NN_IS_PARAMS_H_

#include <stdint.h>
#include "nn_is_size.h"
#include "addr_map/tile_addr_map.h"

/* ==========================================================================
 * Derived geometry
 * ========================================================================== */

#define NN_TH                 (NN_M / MESH_Y_TILES)
#define NN_TW                 (NN_N / MESH_X_TILES)
#define NN_TS                 (NN_K / NN_TIMESLOTS)
#define NN_PB                 (NN_K / MESH_X_TILES)

/* ==========================================================================
 * Phase pipelining
 *
 * Timeslot i finalises Z columns [i*TS, (i+1)*TS) for the whole mesh row, so a
 * PB-wide column block is complete after NN_BLOCK_TS timeslots, and the GEMM
 * splits into exactly MESH_X_TILES rounds:
 *
 *     NN_ROUNDS = NN_TIMESLOTS / (NN_PB / NN_TS) == MESH_X_TILES
 *
 * Round r produces block r, which becomes safe to read NN_PIPE_LAG rounds later
 * (see below). The four tiles of a mesh row then each take a quarter of it *by
 * rows* -- NN_CH rows by the full NN_PB columns -- and post-process it while a
 * later round's GEMM is still running.
 *
 * Splitting by rows and not by columns is the point: LayerNorm is row-wise over
 * PB columns, so every row stays intact and the global result is bit-identical
 * to the non-pipelined version. Splitting by columns would have changed the
 * normalisation groups. The per-tile checksum does change, because a tile now
 * owns a different set of elements.
 * ========================================================================== */

#define NN_BLOCK_TS           (NN_PB / NN_TS)              /* timeslots per column block */
#define NN_ROUNDS             (NN_TIMESLOTS / NN_BLOCK_TS) /* == MESH_X_TILES            */
#define NN_CH                 (NN_TH / MESH_X_TILES)       /* chunk rows owned per round  */

/* How many rounds a tile must have finished past round r before block r is
 * guaranteed final in L2 -- with NO extra barrier.
 *
 * A row barrier per round would be the obvious way to know, and it would be a
 * bad way: it forces the systolic row pipeline to drain NN_ROUNDS times, putting
 * the fill/drain overhead back at (MESH_X-1)/NN_BLOCK_TS instead of
 * (MESH_X-1)/NN_TIMESLOTS -- i.e. undoing exactly what a large NN_TIMESLOTS
 * buys. Avoiding it is necessary but, as measured, not sufficient: the pipeline
 * still drains at each boundary simply because the tile stops advancing the GEMM
 * while it finishes a chunk. Skipping the barrier avoids paying for that twice.
 *
 * The neighbour syncs already carry the ordering we need:
 *
 *   sync_right(j) on tile x pairs with sync_left(j) on tile x+1, and tile x+1
 *   only reaches sync_left(j+1) after completing its own sync_right(j). So
 *   "tile x completed timeslot j" implies "tile x+1 completed timeslot j-1",
 *   and by induction "tile x+k completed timeslot j-k". The rightmost tile --
 *   the one that stores Z to L2 -- is therefore done with timeslot
 *   j-(MESH_X_TILES-1).
 *
 * Block r needs the rightmost tile to have finished timeslot
 * (r+1)*NN_BLOCK_TS-1, so a tile may read it once it has itself completed
 * timeslot (r+1)*NN_BLOCK_TS-1 + (MESH_X_TILES-1), i.e. round r + NN_PIPE_LAG.
 *
 * The golden check validates this: gathering a block early would see Z only
 * partially accumulated, and verify_chunk() would fail loudly. */
#define NN_PIPE_LAG           ((MESH_X_TILES - 1 + NN_BLOCK_TS - 1) / NN_BLOCK_TS)

/* Rows of the chunk handled by each PULP core (SPMD split). */
#define NN_PULP_ROWS_PER_CORE (NN_CH / PULP_CORE_COUNT)

/* Compile-time invariants: a violated one makes an array size negative, so the
 * build stops here rather than hanging or producing garbage at run time. */
typedef char nn_ct_rounds[(NN_TIMESLOTS == NN_ROUNDS * NN_BLOCK_TS) ? 1 : -1];
typedef char nn_ct_mesh[(NN_ROUNDS == MESH_X_TILES) ? 1 : -1];
typedef char nn_ct_chunk[(NN_TH == NN_CH * MESH_X_TILES) ? 1 : -1];
typedef char nn_ct_pulp[(NN_CH == NN_PULP_ROWS_PER_CORE * PULP_CORE_COUNT) ? 1 : -1];
typedef char nn_ct_lag[(NN_PIPE_LAG < NN_ROUNDS) ? 1 : -1];

/* ==========================================================================
 * L1 layout, per tile
 *
 * Absolute addresses: the tile L1 is mapped globally at
 * L1_BASE + tile_id * L1_TILE_OFFSET, and Spatz / the PULP cores reach it
 * through the same OBI crossbar as the CV32 -- so every pointer handed to them
 * must already carry the per-tile base. Fixed 128 KiB slots keep the map valid
 * up to M=N=K=512 on a 4x4 mesh (largest block is 32 KiB).
 * ========================================================================== */

#define NN_L1_SLOT      (0x20000u) /* 128 KiB */

#define NN_L1_OFF_X     (0u * NN_L1_SLOT) /* fp16 [NN_TH x NN_TW] stationary  */
#define NN_L1_OFF_W     (1u * NN_L1_SLOT) /* fp16 [NN_TW x NN_TS] streamed    */
#define NN_L1_OFF_Z0    (2u * NN_L1_SLOT) /* fp16 [NN_TH x NN_TS] accum buf 0 */
#define NN_L1_OFF_Z1    (3u * NN_L1_SLOT) /* fp16 [NN_TH x NN_TS] accum buf 1 */
#define NN_L1_OFF_P     (4u * NN_L1_SLOT) /* fp16  [NN_CH x NN_PB] work chunk */
#define NN_L1_OFF_Q     (5u * NN_L1_SLOT) /* int16 [NN_CH x NN_PB] Q0.15 out  */
#define NN_L1_OFF_CTL   (6u * NN_L1_SLOT) /* param blocks + task results      */

/* One chunk at a time is enough for P and Q: a chunk is always drained and
 * stored before the next is gathered. P is borrowed once more after the timed
 * region, to hold the [NN_CH x NN_K] band pulled back from L2 for the
 * fingerprint -- 32 KiB at M=N=K=512, still well inside its slot. */

/* Control-block sub-offsets inside the NN_L1_OFF_CTL slot. */
#define NN_CTL_OFF_ACT  (0x000u) /* nn_spatz_act_params_t */
#define NN_CTL_OFF_LN   (0x040u) /* nn_spatz_ln_params_t  */
#define NN_CTL_OFF_PULP (0x080u) /* nn_pulp_params_t      */
#define NN_CTL_OFF_RES  (0x100u) /* nn_pulp_core_res_t[PULP_CORE_COUNT] */

/* ==========================================================================
 * L2 layout shared with the VFIO host
 *
 * Sits well below the linked image (text at 0xCC000000, data at 0xCC010000) so
 * the host can DMA into it without touching the ELF. Offsets from the physical
 * L2 origin 0xC0000000 are what the BAR0 DMA engine wants as destination.
 * ========================================================================== */

#define NN_L2_ORIGIN    (0xC0000000u)
#define NN_SHM_BASE     (0xC8000000u)
#define NN_SHM_L2_OFF   (NN_SHM_BASE - NN_L2_ORIGIN) /* 0x08000000 */

#define NN_HDR_ADDR     (NN_SHM_BASE + 0x00000000u) /* nn_hdr_t, 64 B         */
#define NN_X_ADDR       (NN_SHM_BASE + 0x00001000u) /* fp16 [NN_M x NN_N]     */
#define NN_W_ADDR       (NN_X_ADDR + 0x00080000u)   /* fp16 [NN_N x NN_K]     */
#define NN_Y_ADDR       (NN_W_ADDR + 0x00080000u)   /* fp16 [NN_M x NN_K] i/o */
#define NN_Q_ADDR       (NN_Y_ADDR + 0x00080000u)   /* int16 [NN_M x NN_K]    */
#define NN_STAT_ADDR    (NN_Q_ADDR + 0x00080000u)   /* nn_tile_stats_t[16]    */

#define NN_MAGIC        (0x4D414731u) /* "MAG1" -- host-provided payload present */

typedef struct {
    uint32_t magic;   /* NN_MAGIC when the host has staged a payload   */
    uint32_t version; /* contract version, currently 1                */
    uint32_t iter_id; /* host iteration index, echoed back in stats   */
    uint32_t flags;   /* NN_FLAG_*                                    */
    uint32_t m;       /* geometry the host staged, cross-checked here  */
    uint32_t n;
    uint32_t k;
    uint32_t timeslots;
    uint32_t rsvd[8];
} nn_hdr_t; /* 64 B */

#define NN_FLAG_VERIFY  (1u << 0) /* run the golden-sample check          */
#define NN_FLAG_VERBOSE (1u << 1) /* per-tile printf instead of tile 0 only */

typedef struct {
    uint32_t hartid;
    uint32_t iter_id;
    uint32_t cyc_total;
    uint32_t cyc_dma;
    uint32_t cyc_gemm;
    uint32_t cyc_spatz;
    uint32_t cyc_pulp;
    uint32_t cyc_sync;
    uint32_t checksum; /* fp16-bit checksum of the tile's P block */
    uint32_t argmax;   /* argmax reported by PULP core 0          */
    uint32_t errors;
    uint32_t samples_checked;
    uint32_t worst_dev;   /* worst fraction of a sample's fp16 error budget used,
                           * in per-mille; 1000 == exactly at the bound */
    uint32_t cyc_verify;  /* golden check: debug only, excluded from cyc_total */
    uint32_t cyc_post;    /* post-processing time NOT hidden under the GEMM     */
    uint32_t cyc_overlap; /* sum(engine busy) - cyc_total; >0 proves overlap    */
} nn_tile_stats_t;        /* 64 B */

/* ==========================================================================
 * Spatz task parameter blocks
 * ========================================================================== */

typedef struct {
    uintptr_t src;  /* fp16 in                       */
    uintptr_t dst;  /* fp16 out (may alias src)      */
    uint32_t len;   /* element count, multiple of 8   */
    uint16_t scale; /* fp16 bit pattern              */
    uint16_t bias;  /* fp16 bit pattern              */
} nn_spatz_act_params_t;

typedef struct {
    uintptr_t src; /* fp16 in/out, row-major        */
    uint32_t rows;
    uint32_t cols;
    uint16_t gamma;    /* fp16 bit pattern           */
    uint16_t beta;     /* fp16 bit pattern           */
    uint16_t eps;      /* fp16 bit pattern           */
    uint16_t inv_cols; /* fp16 bits of 1/cols: keeps the integer-to-fp16
                        * conversion out of the per-row inner loop */
} nn_spatz_ln_params_t;

/* ==========================================================================
 * PULP cluster task parameter block
 *
 * Pure SPMD: core i owns rows [i*rows_per_core, (i+1)*rows_per_core) and never
 * touches another core's rows or result slot. No inter-core barrier is needed
 * -- the join is the hardware one, ClusterRegs counting NB_CORES_TO_WAIT
 * PULP_DONE writes and raising a single event to the CV32, which is parked in
 * WFE inside eu_pulp_wait().
 * ========================================================================== */

typedef struct {
    uintptr_t src; /* fp16  [rows x cols] in            */
    uintptr_t dst; /* int16 [rows x cols] out, Q0.15    */
    uintptr_t res; /* nn_pulp_core_res_t[PULP_CORE_COUNT] */
    uint32_t rows;
    uint32_t cols; /* multiple of 2: SIMD works on int16 pairs */
    uint32_t rows_per_core;
} nn_pulp_params_t;

typedef struct {
    uint32_t argmax;    /* flat index (within the block) of the core's max */
    int32_t maxval;     /* that maximum, Q8.8                             */
    uint32_t sumsq;     /* sum of squared Q0.15 probabilities, >> 15      */
    uint32_t rows_done; /* progress marker, == rows_per_core on success    */
} nn_pulp_core_res_t;   /* 16 B */

#endif /* NN_IS_PARAMS_H_ */
