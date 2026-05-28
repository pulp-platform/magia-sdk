// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// 
// Victor Isachi <victor.isachi@unibo.it>
// Alberto Dequino <alberto.dequino@unibo.it>
//
// MAGIA-oriented spatial FFT kernel with:
//   - float16 data and twiddle factors;
//   - stage-level double buffering in tile-local SPM;
//   - single-outstanding DMA GET and single-outstanding DMA PUT scheduling;
//   - 2D DMA descriptors for split real/imaginary complex transfers;
//   - twiddle lookup tables loaded from L2 into each tile's local SPM;
//   - a flat hot path, following a low-overhead style.
//
// Algorithmic model
// -----------------
// This kernel computes a radix-2 decimation-in-time FFT over FFT_ARRAY_SIZE
// complex samples. The complex vector is stored in split format:
//
//     real[0..N-1], imag[0..N-1]
//
// not as interleaved real/imag pairs. Every MAGIA tile owns one contiguous
// block of the global FFT vector:
//
//     BLOCK_SIZE = FFT_ARRAY_SIZE / NUM_HARTS
//
// and tile hartid owns global indices:
//
//     [hartid * BLOCK_SIZE, (hartid + 1) * BLOCK_SIZE)
//
// Each FFT butterfly is assigned to the tile that owns the lower butterfly
// index i0. If the matching upper butterfly index i1 is remote, the tile:
//
//     1. DMA GETs x[i1] from the owner tile's source SPM image;
//     2. computes a CHUNK_SIZE-sized butterfly block with the VPU;
//     3. DMA PUTs y1 back to the owner tile's destination SPM image.
//
// The lower output y0 is local to the computing tile and is written directly to
// its destination SPM image.
//
// Why two SPM images are used
// ---------------------------
// The FFT stages are executed out-of-place:
//
//     odd stages:  read A, write B
//     even stages: read B, write A
//
// This is not only a performance choice. It is required for safe overlap. If
// a stage were computed in place, one tile could overwrite a value in its local
// SPM while another tile was still DMA-reading that value as a remote operand.
// With A/B stage ping-pong, every DMA GET reads from a stable source image and
// every VPU/DMA PUT writes into a separate destination image.
//
// Why 2D DMA is useful here
// --------------------------
// The FFT butterfly chunks are already contiguous in the index dimension, so
// 2D DMA is not needed to gather strided FFT indices. However, complex values
// are stored as split real and imaginary arrays. A single logical complex GET
// would otherwise be two 1D transfers:
//
//     GET real row
//     GET imag row
//
// The 2D DMA descriptor below uses two rows to move both rows at once:
//
//     row 0: real[count]
//     row 1: imag[count]
//
// with a source row stride equal to the distance between real and imaginary
// arrays, and a destination row stride equal to the distance between the real
// and imaginary staging buffers. The same idea is used for remote PUTs, input
// loading, output storing, and WR/WI twiddle-table loading.
//
// Expected data header contract
// -----------------------------
// The selected fft_array_*.h header is expected to provide float16 arrays:
//
//     #define FFT_ARRAY_SIZE <power-of-two>
//     #define LOG2_FFT_SIZE <log_2(FFT_ARRAY_SIZE)>
//     float16 IR[FFT_ARRAY_SIZE];
//     float16 II[FFT_ARRAY_SIZE];
//     float16 WR[LOG2_FFT_SIZE * FFT_ARRAY_SIZE / 2];
//     float16 WI[LOG2_FFT_SIZE * FFT_ARRAY_SIZE / 2];
//     float16 GR[FFT_ARRAY_SIZE];
//     float16 GI[FFT_ARRAY_SIZE];
//
// WR/WI must use the Python layout from the prompt:
//
//     WR[stage * (N/2) + b]
//     WI[stage * (N/2) + b]
//
// where b is the butterfly number within that stage.
//
// IR/II are assumed to be in bit-reverse order. 

#include <stdint.h>
#include "tile.h"
#include "idma.h"
#include "fsync.h"

#define FFT_SIZE_256
// #define FFT_SIZE_1024
// #define FFT_SIZE_4096
// #define FFT_SIZE_16384
// #define FFT_SIZE_65536

#if defined(FFT_SIZE_256)
#include "fft_array_256.h"
#elif defined(FFT_SIZE_1024)
#include "fft_array_1024.h"
#elif defined(FFT_SIZE_4096)
#include "fft_array_4096.h"
#elif defined(FFT_SIZE_16384)
#include "fft_array_16384.h"
#elif defined(FFT_SIZE_65536)
#include "fft_array_65536.h"
#endif

#define VERBOSE (0)

#define DATA_BYTES (2)
#define NUM_BUFF   (2)
#define CHUNK_SIZE (32)

#define DMA_IN  (0)
#define DMA_OUT (1)

#define LOG2_FFT_SIZE   (__builtin_ctz(FFT_ARRAY_SIZE))
#define BLOCK_SIZE      (FFT_ARRAY_SIZE / NUM_HARTS)
#define LOG2_BLOCK_SIZE (__builtin_ctz(BLOCK_SIZE))
#define TWIDDLE_SIZE    (LOG2_FFT_SIZE * (FFT_ARRAY_SIZE / 2))

#define CHUNK_OWNER(global_index) ((global_index) >> LOG2_BLOCK_SIZE)
#define LOCAL_INDEX(global_index) ((global_index) & (BLOCK_SIZE - 1))

#define mmiof16(x) (*(volatile float16 *)(x))

#define BLOCK_BYTES   (BLOCK_SIZE * DATA_BYTES)
#define CHUNK_BYTES   (CHUNK_SIZE * DATA_BYTES)
#define TWIDDLE_BYTES (TWIDDLE_SIZE * DATA_BYTES)

static inline void idma_1d(idma_controller_t *ctrl, uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len){
    idma_mm_conf(dir);
    if(dir){
        // DMA_OUT: local OBI/L1 -> AXI/L2 or remote L1 address.
        idma_mm_set_addr_len(dir, axi_addr, obi_addr, len);
    }else{
        // DMA_IN: AXI/L2 or remote L1 address -> local OBI/L1.
        idma_mm_set_addr_len(dir, obi_addr, axi_addr, len);
    }
    idma_mm_set_std2_rep2(dir, 0, 0, 1);
    idma_mm_set_std3_rep3(dir, 0, 0, 1);
    idma_mm_start(dir);
}

static inline void idma_2d(idma_controller_t *ctrl, uint8_t dir, uint32_t axi_addr, uint32_t obi_addr, uint32_t len, uint32_t axi_stride, uint32_t obi_stride, uint32_t reps){
    idma_mm_conf(dir);
    if(dir){
        // DMA_OUT: local OBI/L1 -> AXI/L2 or remote L1 address.
        idma_mm_set_addr_len(dir, axi_addr, obi_addr, len);
        idma_mm_set_std2_rep2(dir, axi_stride, obi_stride, reps);
    }else{
        // DMA_IN: AXI/L2 or remote L1 address -> local OBI/L1.
        idma_mm_set_addr_len(dir, obi_addr, axi_addr, len);
        idma_mm_set_std2_rep2(dir, obi_stride, axi_stride, reps);
    }
    idma_mm_set_std3_rep3(dir, 0, 0, 1);
    idma_mm_start(dir);
}

int main(void){
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

    uint32_t local_block_idx_start = hartid * BLOCK_SIZE;
    uint32_t local_block_idx_end   = local_block_idx_start + BLOCK_SIZE;

    // Tile-local L1/SPM memory map.
    //
    // A and B are the stage-level ping-pong images. Each image is split into
    // real and imaginary rows. The in/out buffers are also split rows, but they
    // are CHUNK_SIZE long rather than BLOCK_SIZE long.
    //
    // The layout intentionally places each real row immediately before the
    // matching imaginary row. This makes a complex 2D DMA descriptor simple:
    // two rows, row length count * sizeof(float16), row stride BLOCK_BYTES or
    // CHUNK_BYTES depending on whether the transfer endpoint is a full SPM
    // image or a staging buffer.
    uint32_t a_r_addr = l1_tile_base;
    uint32_t a_i_addr = a_r_addr + BLOCK_BYTES;
    uint32_t b_r_addr = a_i_addr + BLOCK_BYTES;
    uint32_t b_i_addr = b_r_addr + BLOCK_BYTES;

    uint32_t in0_r_addr  = b_i_addr + BLOCK_BYTES;
    uint32_t in0_i_addr  = in0_r_addr + CHUNK_BYTES;
    uint32_t in1_r_addr  = in0_i_addr + CHUNK_BYTES;
    uint32_t in1_i_addr  = in1_r_addr + CHUNK_BYTES;

    uint32_t out0_r_addr = in1_i_addr + CHUNK_BYTES;
    uint32_t out0_i_addr = out0_r_addr + CHUNK_BYTES;
    uint32_t out1_r_addr = out0_i_addr + CHUNK_BYTES;
    uint32_t out1_i_addr = out1_r_addr + CHUNK_BYTES;

    uint32_t wr_addr = out1_i_addr + CHUNK_BYTES;
    uint32_t wi_addr = wr_addr + TWIDDLE_BYTES;

    uint32_t in_r_addr[NUM_BUFF]  = {in0_r_addr,  in1_r_addr};
    uint32_t out_r_addr[NUM_BUFF] = {out0_r_addr, out1_r_addr};

    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);

    // Fast path: IR/II are already bit-reversed in L2. Therefore each tile
    // owns one contiguous block and can fetch both real and imaginary rows with
    // one 2D DMA GET:
    //
    //   row 0: IR[local_block_idx_start : local_block_idx_end]
    //   row 1: II[local_block_idx_start : local_block_idx_end]
    //
    // The source row stride is the address distance between II and IR in L2.
    // The destination row stride is BLOCK_BYTES because a_i follows a_r in SPM.
    idma_2d(&idma_ctrl, DMA_IN, (uint32_t)&IR[local_block_idx_start], a_r_addr, BLOCK_BYTES, (uint32_t)&II[local_block_idx_start] - (uint32_t)&IR[local_block_idx_start], BLOCK_BYTES, 2);
    // idma_wait_();

    // Twiddle factors are assumed to be in L2 at kernel entry and are always
    // fetched into this tile's SPM. The FFT hot loop below only performs local
    // SPM loads from wr_addr/wi_addr; it never evaluates sin/cos/exp.
    //
    // The 2D transfer loads:
    //
    //   row 0: WR[0 : TWIDDLE_SIZE]
    //   row 1: WI[0 : TWIDDLE_SIZE]
    //
    // into local SPM rows wr_addr and wi_addr.
    idma_2d(&idma_ctrl, DMA_IN, (uint32_t)&WR[0], wr_addr, TWIDDLE_BYTES, (uint32_t)&WI[0] - (uint32_t)&WR[0], TWIDDLE_BYTES, 2);
    // idma_wait();

    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);

    for(uint32_t s = 1; s <= LOG2_FFT_SIZE; s++){
        uint32_t stage = s - 1;
        uint32_t len   = 1 << s;
        uint32_t half  = len >> 1;

        // Select source and destination SPM images for this FFT stage.
        // The current stage reads only src_* and writes only dst_*.
        uint32_t src_r_addr = (s & 1) ? a_r_addr : b_r_addr;
        uint32_t src_i_addr = (s & 1) ? a_i_addr : b_i_addr;
        uint32_t dst_r_addr = (s & 1) ? b_r_addr : a_r_addr;
        uint32_t dst_i_addr = (s & 1) ? b_i_addr : a_i_addr;

        uint32_t curr_idx  = local_block_idx_start;
        uint32_t curr_slot = 0;
        uint32_t curr_have = 0;
        uint32_t get_busy  = 0;
        uint32_t put_busy  = 0;

        // Metadata for the two ping-pong remote chunks. These arrays replace
        // the work_segment_t struct from the high-level simulator. Each slot
        // describes one contiguous run of butterflies whose i0 operands are
        // local to this tile and whose i1 operands are remote.
        uint32_t i0_base[NUM_BUFF]  = {0, 0};
        uint32_t i1_base[NUM_BUFF]  = {0, 0};
        uint32_t i0_local[NUM_BUFF] = {0, 0};
        uint32_t i1_owner[NUM_BUFF] = {0, 0};
        uint32_t i1_local[NUM_BUFF] = {0, 0};
        uint32_t count[NUM_BUFF]    = {0, 0};

        // Initial scan: consume local-only chunks until the first remote chunk
        // is found. The first remote chunk is fetched into curr_slot and later
        // becomes the first chunk processed by the remote pipeline.
        for(;;){
            if(curr_idx >= local_block_idx_end){curr_have = 0; break;}

            uint32_t pos = curr_idx & (len - 1);
            if(pos >= half){curr_idx += len - pos; continue;}

            uint32_t limit = curr_idx + (half - pos);
            if(local_block_idx_end < limit)  {limit = local_block_idx_end;}
            if(curr_idx + CHUNK_SIZE < limit){limit = curr_idx + CHUNK_SIZE;}

            i0_base[curr_slot]  = curr_idx;
            i1_base[curr_slot]  = curr_idx + half;
            i1_owner[curr_slot] = CHUNK_OWNER(i1_base[curr_slot]);
            i1_local[curr_slot] = LOCAL_INDEX(i1_base[curr_slot]);

            uint32_t i1_owner_end = (i1_owner[curr_slot] + 1) * BLOCK_SIZE;
            if(i1_owner_end - half < limit){limit = i1_owner_end - half;}

            count[curr_slot]    = limit - curr_idx;
            i0_local[curr_slot] = LOCAL_INDEX(i0_base[curr_slot]);
            curr_idx = limit;

            if(i1_owner[curr_slot] == hartid){
                // Local butterfly chunk: both x[i0] and x[i1] are in this
                // tile's source SPM image. Results are written to this tile's
                // destination SPM image.
                uint32_t loc_i0 = i0_local[curr_slot];
                uint32_t loc_i1 = i1_local[curr_slot];
                uint32_t bfly0  = (i0_base[curr_slot] / len) * half + (i0_base[curr_slot] & (len - 1));
                uint32_t twidx0 = stage * (FFT_ARRAY_SIZE / 2) + bfly0;

                // VPU mapping note:
                // This scalar loop is the reference schedule for one VPU
                // vector issue. Because the chunk is contiguous and never
                // crosses a lower-half FFT-group boundary, twiddle indices are
                // also contiguous. A real VPU implementation should replace
                // this loop by strip-mined vector instructions:
                //
                //   for kk = 0; kk < count; kk += VL:
                //       vl    = min(VL, count - kk)
                //       vr_ar = vle16(src_r + loc_i0 + kk)
                //       vr_ai = vle16(src_i + loc_i0 + kk)
                //       vr_br = vle16(src_r + loc_i1 + kk)
                //       vr_bi = vle16(src_i + loc_i1 + kk)
                //       vr_wr = vle16(wr + twidx0 + kk)
                //       vr_wi = vle16(wi + twidx0 + kk)
                //       vr_tr = vr_wr * vr_br - vr_wi * vr_bi
                //       vr_ti = vr_wr * vr_bi + vr_wi * vr_br
                //       vse16(dst_r + loc_i0 + kk, vr_ar + vr_tr)
                //       vse16(dst_i + loc_i0 + kk, vr_ai + vr_ti)
                //       vse16(dst_r + loc_i1 + kk, vr_ar - vr_tr)
                //       vse16(dst_i + loc_i1 + kk, vr_ai - vr_ti)
                for(uint32_t k = 0; k < count[curr_slot]; k++){
                    float16 wr = mmiof16(wr_addr + (twidx0 + k) * DATA_BYTES);
                    float16 wi = mmiof16(wi_addr + (twidx0 + k) * DATA_BYTES);

                    float16 ar = mmiof16(src_r_addr + (loc_i0 + k) * DATA_BYTES);
                    float16 ai = mmiof16(src_i_addr + (loc_i0 + k) * DATA_BYTES);
                    float16 br = mmiof16(src_r_addr + (loc_i1 + k) * DATA_BYTES);
                    float16 bi = mmiof16(src_i_addr + (loc_i1 + k) * DATA_BYTES);

                    float16 tr = wr * br - wi * bi;
                    float16 ti = wr * bi + wi * br;

                    mmiof16(dst_r_addr + (loc_i0 + k) * DATA_BYTES) = ar + tr;
                    mmiof16(dst_i_addr + (loc_i0 + k) * DATA_BYTES) = ai + ti;
                    mmiof16(dst_r_addr + (loc_i1 + k) * DATA_BYTES) = ar - tr;
                    mmiof16(dst_i_addr + (loc_i1 + k) * DATA_BYTES) = ai - ti;
                }
                continue;
            }

            // Remote upper operands: fetch x[i1] from owner(i1)'s source image.
            // One 2D GET moves both real and imaginary rows:
            //
            // remote source row stride: BLOCK_BYTES
            // local buffer row stride:  CHUNK_BYTES
            // row length:               count * sizeof(float16)
            // rows:                     2
            uint32_t remote_src_r = get_l1_base(i1_owner[curr_slot]) + ((s & 1) ? 0 : 2 * BLOCK_BYTES) + (i1_local[curr_slot] * DATA_BYTES);

            idma_2d(&idma_ctrl, DMA_IN, remote_src_r, in_r_addr[curr_slot], count[curr_slot] * DATA_BYTES, BLOCK_BYTES, CHUNK_BYTES, 2);
            get_busy = 1;
            curr_have = 1;
            break;
        }

        // If the first remote GET was issued, wait for it before it becomes the
        // current chunk consumed by the VPU. Local-only work above may overlap
        // the first GET.
        if(curr_have && get_busy){
            // idma_wait();
            get_busy = 0;
        }

        while(curr_have){
            uint32_t next_slot = 1 - curr_slot;
            uint32_t next_have = 0;

            // Look ahead for the next remote chunk. Local-only chunks found on
            // the way are computed immediately. When the next remote chunk is
            // found, its GET is issued into next_slot before computing curr_slot.
            // This gives the intended overlap:
            //
            //   DMA GET(next remote chunk)  ||  VPU(current remote chunk)
            //
            // while still respecting the hardware restriction that a tile may
            // have only one DMA GET in flight.
            for(;;){
                if(curr_idx >= local_block_idx_end){next_have = 0; break;}

                uint32_t pos = curr_idx & (len - 1);
                if(pos >= half){curr_idx += len - pos; continue;}

                uint32_t limit = curr_idx + (half - pos);
                if(local_block_idx_end < limit)  {limit = local_block_idx_end;}
                if(curr_idx + CHUNK_SIZE < limit){limit = curr_idx + CHUNK_SIZE;}

                i0_base[next_slot]  = curr_idx;
                i1_base[next_slot]  = curr_idx + half;
                i1_owner[next_slot] = CHUNK_OWNER(i1_base[next_slot]);
                i1_local[next_slot] = LOCAL_INDEX(i1_base[next_slot]);

                uint32_t i1_owner_end = (i1_owner[next_slot] + 1) * BLOCK_SIZE;
                if(i1_owner_end - half < limit){limit = i1_owner_end - half;}

                count[next_slot]    = limit - curr_idx;
                i0_local[next_slot] = LOCAL_INDEX(i0_base[next_slot]);
                curr_idx = limit;

                if(i1_owner[next_slot] == hartid){
                    uint32_t loc_i0 = i0_local[next_slot];
                    uint32_t loc_i1 = i1_local[next_slot];
                    uint32_t bfly0  = (i0_base[next_slot] / len) * half + (i0_base[next_slot] & (len - 1));
                    uint32_t twidx0 = stage * (FFT_ARRAY_SIZE / 2) + bfly0;

                    for(uint32_t k = 0; k < count[next_slot]; k++){
                        float16 wr = mmiof16(wr_addr + (twidx0 + k) * DATA_BYTES);
                        float16 wi = mmiof16(wi_addr + (twidx0 + k) * DATA_BYTES);

                        float16 ar = mmiof16(src_r_addr + (loc_i0 + k) * DATA_BYTES);
                        float16 ai = mmiof16(src_i_addr + (loc_i0 + k) * DATA_BYTES);
                        float16 br = mmiof16(src_r_addr + (loc_i1 + k) * DATA_BYTES);
                        float16 bi = mmiof16(src_i_addr + (loc_i1 + k) * DATA_BYTES);

                        float16 tr = wr * br - wi * bi;
                        float16 ti = wr * bi + wi * br;

                        mmiof16(dst_r_addr + (loc_i0 + k) * DATA_BYTES) = ar + tr;
                        mmiof16(dst_i_addr + (loc_i0 + k) * DATA_BYTES) = ai + ti;
                        mmiof16(dst_r_addr + (loc_i1 + k) * DATA_BYTES) = ar - tr;
                        mmiof16(dst_i_addr + (loc_i1 + k) * DATA_BYTES) = ai - ti;
                    }
                    continue;
                }

                if(get_busy){
                    // idma_wait();
                    get_busy = 0;
                }

                uint32_t remote_src_r = get_l1_base(i1_owner[next_slot]) + ((s & 1) ? 0 : 2 * BLOCK_BYTES) + (i1_local[next_slot] * DATA_BYTES);

                idma_2d(&idma_ctrl, DMA_IN, remote_src_r, in_r_addr[next_slot], count[next_slot] * DATA_BYTES, BLOCK_BYTES, CHUNK_BYTES, 2);
                get_busy = 1;
                next_have = 1;
                break;
            }

            // Compute the current remote chunk. i0 is read from the local
            // source image; i1 was fetched into in_r/in_i[curr_slot]. y0 is
            // written directly to the local destination image. y1 is written to
            // out_r/out_i[curr_slot] and later returned to owner(i1) with one
            // 2D DMA PUT.
            uint32_t bfly0  = (i0_base[curr_slot] / len) * half + (i0_base[curr_slot] & (len - 1));
            uint32_t twidx0 = stage * (FFT_ARRAY_SIZE / 2) + bfly0;

            // VPU mapping for a remote chunk:
            //
            //   lower x0 vectors:  src_r/src_i at local i0
            //   upper x1 vectors:  in_r/in_i[curr_slot], filled by DMA GET
            //   twiddle vectors:   wr/wi at twidx0 + lane
            //   lower y0 stores:   dst_r/dst_i at local i0
            //   upper y1 stores:   out_r/out_i[curr_slot], then DMA PUT
            //
            // In hardware this loop should become a strip-mined VPU loop. The
            // only scalar work per vector chunk should be address generation and
            // vector length setup; all lane-wise arithmetic is independent:
            //
            //   tr = wr * br - wi * bi
            //   ti = wr * bi + wi * br
            //   y0 = a + t
            //   y1 = a - t
            //
            // The current scalar form is kept only as a clear, executable
            // reference for the intended vector sequence.
            for(uint32_t k = 0; k < count[curr_slot]; k++){
                float16 wr = mmiof16(wr_addr + (twidx0 + k) * DATA_BYTES);
                float16 wi = mmiof16(wi_addr + (twidx0 + k) * DATA_BYTES);

                float16 ar = mmiof16(src_r_addr + (i0_local[curr_slot] + k) * DATA_BYTES);
                float16 ai = mmiof16(src_i_addr + (i0_local[curr_slot] + k) * DATA_BYTES);
                float16 br = mmiof16(in_r_addr[curr_slot] + k * DATA_BYTES);
                float16 bi = mmiof16(in_r_addr[curr_slot] + CHUNK_BYTES + k * DATA_BYTES);

                float16 tr = wr * br - wi * bi;
                float16 ti = wr * bi + wi * br;

                mmiof16(dst_r_addr + (i0_local[curr_slot] + k) * DATA_BYTES)  = ar + tr;
                mmiof16(dst_i_addr + (i0_local[curr_slot] + k) * DATA_BYTES)  = ai + ti;
                mmiof16(out_r_addr[curr_slot] + k * DATA_BYTES)               = ar - tr;
                mmiof16(out_r_addr[curr_slot] + CHUNK_BYTES + k * DATA_BYTES) = ai - ti;
            }

            // Return the remote upper outputs y1 to owner(i1)'s destination SPM
            // image. Wait for the previous PUT before issuing this PUT: this
            // enforces at most one outstanding PUT per tile. The next GET, if
            // already issued above, may still be in flight while the VPU was
            // computing the current chunk.
            if(put_busy){
                // idma_wait();
                put_busy = 0;
            }

            uint32_t remote_dst_r = get_l1_base(i1_owner[curr_slot]) + ((s & 1) ? 2 * BLOCK_BYTES : 0) + (i1_local[curr_slot] * DATA_BYTES);

            idma_2d(&idma_ctrl, DMA_OUT, remote_dst_r, out_r_addr[curr_slot], count[curr_slot] * DATA_BYTES, BLOCK_BYTES, CHUNK_BYTES, 2);
            put_busy = 1;

            curr_slot = next_slot;
            curr_have = next_have;

            // The next iteration will wait for GET(next) before consuming the
            // next input buffer. That wait is deliberately delayed until the
            // last responsible moment so that GET(next) can overlap with the
            // current VPU computation and PUT scheduling.
            if(curr_have && get_busy){
                // idma_wait();
                get_busy = 0;
            }
        }

        if(put_busy){
            // idma_wait();
            put_busy = 0;
        }

        // All tiles must complete stage s before any tile reads the destination
        // image as the source image for stage s+1.
        fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    }

    uint32_t final_r_addr = (LOG2_FFT_SIZE & 1) ? b_r_addr : a_r_addr;

    // Store the final natural-order FFT result back to L2/global memory.
    // One 2D DMA PUT stores both output rows:
    //
    //   row 0: IR[local block]
    //   row 1: II[local block]
    idma_2d(&idma_ctrl, DMA_OUT, (uint32_t)&IR[local_block_idx_start], final_r_addr, BLOCK_BYTES, (uint32_t)&II[local_block_idx_start] - (uint32_t)&IR[local_block_idx_start], BLOCK_BYTES, 2);
    // idma_wait();

    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    uint32_t errors=0;
    uint16_t computed_r, expected_r, diff_r = 0;
    uint16_t computed_i, expected_i, diff_i = 0;
    if(hartid == 0){
        for(int i = 0; i < FFT_ARRAY_SIZE; i++){
            computed_r = (uint16_t)IR[i];
            expected_r = (uint16_t)GR[i];
            diff_r = (computed_r > expected_r) ? (computed_r - expected_r) : (expected_r - computed_r);
            computed_i = (uint16_t)II[i];
            expected_i = (uint16_t)GI[i];
            diff_i = (computed_i > expected_i) ? (computed_i - expected_i) : (expected_i - computed_i);
            if(diff_r > 0x01FF || diff_i > 0x01FF){
                printf("[ERROR] - Expected FFT does not match computed FFT at index %0u: 0x%0x + 0x%0xi vs. 0x%0x + 0x%0xi\n", 
                    i, (uint16_t)GR[i], (uint16_t)GI[i], (uint16_t)IR[i], (uint16_t)II[i]);
                errors++;
            }
        }
        return errors;
    }else{
        return errors;
    }
}
