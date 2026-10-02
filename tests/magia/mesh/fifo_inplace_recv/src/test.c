// Copyright 2026 Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * In-place FIFO receives.
 *
 *   tile 0: L2 -> A (8x16)  --tr0--> tile 1 S (whole, dense: in place)
 *                           --tr3--> tile 1 F (whole, but forwarded: unpacked)
 *   tile 2: L2 -> B (8x32)  --tr1--> tile 1 P[:, 0:16]  (strided: unpacked)
 *                           --tr2--> tile 1 P[:, 16:32] (strided: unpacked)
 *   tile 1: copy S -> U; U -> out0; P -> out1;  F --tr4--> tile 3 G
 *   tile 3: G (whole, dense: in place); copy G -> H; H -> out2
 *
 * Every consumer FIFO has a single slot per producer, so each in-place hold
 * directly back-pressures its producer.  The planned storage of S and G is
 * filled with a sentinel: it must stay untouched when receives are consumed
 * in place, and be overwritten when MAPS_FIFO_INPLACE_RECV=0.
 */

#include <stdint.h>

#include "eventunit.h"
#include "fsync.h"
#include "idma.h"
#include "tile.h"
#include "utils/maps_utils_v2.h"
#include "utils/performance_utils.h"
#include "utils/printf.h"

#define NUM_TOKENS   4u
#define TOKEN_SLOTS  2u
#define ROWS         8u
#define COLS         16u
#define WIDE_COLS    32u
#define ELEM         2u
#define NARROW_BYTES (ROWS * COLS * ELEM)
#define WIDE_BYTES   (ROWS * WIDE_COLS * ELEM)
#define DATA_OFFSET  0x00010000u
#define SENTINEL     0xdeadu

enum { SLICE_A, SLICE_B, SLICE_S, SLICE_U, SLICE_P, SLICE_F, SLICE_G, SLICE_H };

static uint16_t input0[NUM_TOKENS][ROWS][COLS]
    __attribute__((section(".l2_bulk.fifo_inplace_in0")));
static uint16_t input1[NUM_TOKENS][ROWS][WIDE_COLS]
    __attribute__((section(".l2_bulk.fifo_inplace_in1")));
static uint16_t output0[NUM_TOKENS][ROWS][COLS]
    __attribute__((section(".l2_bulk.fifo_inplace_out0")));
static uint16_t output1[NUM_TOKENS][ROWS][WIDE_COLS]
    __attribute__((section(".l2_bulk.fifo_inplace_out1")));
static uint16_t output2[NUM_TOKENS][ROWS][COLS]
    __attribute__((section(".l2_bulk.fifo_inplace_out2")));

static slice_desc_t slices[4][4];
static l2_read_desc_t l2_reads[4];
static fifo_recv_desc_t recvs[4][4];
static op_desc_t ops[4];
static fifo_send_desc_t sends[4][2];
static l2_write_desc_t l2_writes[4][2];

static uint16_t input0_value(uint32_t token, uint32_t row, uint32_t col)
{
    return (uint16_t)(0x1000u + token * 0x100u + row * COLS + col);
}

static uint16_t input1_value(uint32_t token, uint32_t row, uint32_t col)
{
    return (uint16_t)(0x8000u + token * 0x200u + row * WIDE_COLS + col);
}

static slice_desc_t matrix_slice(uint32_t id, uint32_t cols, uint32_t l1_offset)
{
    slice_desc_t slice = {
        .slice_id = id,
        .global_id = id,
        .global_kind = GLOBAL_INTERMEDIATE,
        .rank = 2u,
        .shape = {ROWS, cols},
        .elem_type = ELEM_F16,
        .elem_bytes = ELEM,
        .l1_offset_bytes = l1_offset,
        .slot_bytes = ROWS * cols * ELEM,
        .num_slots = TOKEN_SLOTS,
        .strides_bytes = {cols * ELEM, ELEM},
    };
    return slice;
}

/* An 8 x `cols` window starting at column `col` of an 8 x `slice_cols` slice. */
static subslice_desc_t window(uint32_t id, uint32_t slice_cols, uint32_t col, uint32_t cols)
{
    subslice_desc_t sub = {
        .slice_id = id,
        .offset = {0u, col},
        .rank = 2u,
        .shape = {ROWS, cols},
        .elem_type = ELEM_F16,
        .elem_bytes = ELEM,
        .strides_bytes = {slice_cols * ELEM, ELEM},
    };
    return sub;
}

static tensor_sub_slice_t layout(uint32_t row_stride, uint32_t cols)
{
    tensor_sub_slice_t slice = {
        .rank = 2u,
        .num_elems = ROWS * cols,
        .dims = {{0u, ROWS, row_stride}, {0u, cols, ELEM}},
    };
    return slice;
}

static fifo_recv_desc_t recv(uint32_t transition, uint32_t src, uint32_t dst,
                             subslice_desc_t sub, uint32_t producer_idx)
{
    fifo_recv_desc_t desc = {
        .transition_id = transition,
        .src_hartid = src,
        .dst_hartid = dst,
        .dst = sub,
        .producer_idx = producer_idx,
    };
    return desc;
}

static fifo_send_desc_t send(uint32_t transition, uint32_t src, uint32_t dst,
                             subslice_desc_t from, subslice_desc_t to,
                             uint32_t producer_idx)
{
    fifo_send_desc_t desc = {
        .transition_id = transition,
        .src_hartid = src,
        .dst_hartid = dst,
        .src = from,
        .dst = to,
        .copy_src = layout(from.strides_bytes[0], from.shape[1]),
        .copy_dst = layout(to.strides_bytes[0], to.shape[1]),
        .producer_idx = producer_idx,
    };
    return desc;
}

static l2_write_desc_t l2_write(subslice_desc_t from, uint32_t dst, uint32_t cols)
{
    l2_write_desc_t desc = {
        .global_id = from.slice_id,
        .global_kind = GLOBAL_OUTPUT,
        .dst_l2_addr = dst,
        .dst_l2_token_stride_bytes = ROWS * cols * ELEM,
        .src = from,
        .copy_src = layout(cols * ELEM, cols),
        .copy_dst = layout(cols * ELEM, cols),
    };
    return desc;
}

static op_desc_t copy_op(uint32_t from, uint32_t to)
{
    op_desc_t op = {
        .kind = OP_COPY,
        .num_inputs = 1u,
        .inputs = {window(from, COLS, 0u, COLS)},
        .num_outputs = 1u,
        .outputs = {window(to, COLS, 0u, COLS)},
    };
    return op;
}

static void build_plan(uint32_t hartid, fifo_tile_plan_t *plan)
{
    const subslice_desc_t a = window(SLICE_A, COLS, 0u, COLS);
    const subslice_desc_t b_left = window(SLICE_B, WIDE_COLS, 0u, COLS);
    const subslice_desc_t b_right = window(SLICE_B, WIDE_COLS, COLS, COLS);
    const subslice_desc_t s = window(SLICE_S, COLS, 0u, COLS);
    const subslice_desc_t p_left = window(SLICE_P, WIDE_COLS, 0u, COLS);
    const subslice_desc_t p_right = window(SLICE_P, WIDE_COLS, COLS, COLS);
    const subslice_desc_t f = window(SLICE_F, COLS, 0u, COLS);
    const subslice_desc_t g = window(SLICE_G, COLS, 0u, COLS);

    *plan = (fifo_tile_plan_t){
        .hartid = hartid,
        .l1_data_base = get_l1_base(hartid) + DATA_OFFSET,
        .num_token_slots = TOKEN_SLOTS,
        .slices = slices[hartid],
        .l2_reads = &l2_reads[hartid],
        .recvs = recvs[hartid],
        .ops = &ops[hartid],
        .sends = sends[hartid],
        .l2_writes = l2_writes[hartid],
        .fifo = {.num_slots = 1u, .slot_data_size = NARROW_BYTES},
    };

    switch (hartid) {
    case 0u:
        slices[0][0] = matrix_slice(SLICE_A, COLS, 0u);
        plan->num_slices = 1u;
        l2_reads[0] = (l2_read_desc_t){
            .global_id = SLICE_A,
            .src_l2_addr = (uint32_t)input0,
            .src_l2_token_stride_bytes = NARROW_BYTES,
            .dst = a,
            .copy_src = layout(COLS * ELEM, COLS),
            .copy_dst = layout(COLS * ELEM, COLS),
        };
        plan->num_l2_reads = 1u;
        sends[0][0] = send(0u, 0u, 1u, a, s, 0u);
        sends[0][1] = send(3u, 0u, 1u, a, f, 3u);
        plan->num_sends = 2u;
        break;
    case 2u:
        slices[2][0] = matrix_slice(SLICE_B, WIDE_COLS, 0u);
        plan->num_slices = 1u;
        l2_reads[2] = (l2_read_desc_t){
            .global_id = SLICE_B,
            .src_l2_addr = (uint32_t)input1,
            .src_l2_token_stride_bytes = WIDE_BYTES,
            .dst = window(SLICE_B, WIDE_COLS, 0u, WIDE_COLS),
            .copy_src = layout(WIDE_COLS * ELEM, WIDE_COLS),
            .copy_dst = layout(WIDE_COLS * ELEM, WIDE_COLS),
        };
        plan->num_l2_reads = 1u;
        sends[2][0] = send(1u, 2u, 1u, b_left, p_left, 1u);
        sends[2][1] = send(2u, 2u, 1u, b_right, p_right, 2u);
        plan->num_sends = 2u;
        break;
    case 1u:
        slices[1][0] = matrix_slice(SLICE_S, COLS, 0u);
        slices[1][1] = matrix_slice(SLICE_U, COLS, 2u * NARROW_BYTES);
        slices[1][2] = matrix_slice(SLICE_P, WIDE_COLS, 4u * NARROW_BYTES);
        slices[1][3] = matrix_slice(SLICE_F, COLS, 4u * NARROW_BYTES + 2u * WIDE_BYTES);
        plan->num_slices = 4u;
        recvs[1][0] = recv(0u, 0u, 1u, s, 0u);
        recvs[1][1] = recv(1u, 2u, 1u, p_left, 1u);
        recvs[1][2] = recv(2u, 2u, 1u, p_right, 2u);
        recvs[1][3] = recv(3u, 0u, 1u, f, 3u);
        plan->num_recvs = 4u;
        ops[1] = copy_op(SLICE_S, SLICE_U);
        plan->num_ops = 1u;
        sends[1][0] = send(4u, 1u, 3u, f, g, 0u);
        plan->num_sends = 1u;
        l2_writes[1][0] = l2_write(window(SLICE_U, COLS, 0u, COLS), (uint32_t)output0, COLS);
        l2_writes[1][1] = l2_write(window(SLICE_P, WIDE_COLS, 0u, WIDE_COLS),
                                   (uint32_t)output1, WIDE_COLS);
        plan->num_l2_writes = 2u;
        plan->fifo.num_producers = 4u;
        break;
    case 3u:
        slices[3][0] = matrix_slice(SLICE_G, COLS, 0u);
        slices[3][1] = matrix_slice(SLICE_H, COLS, 2u * NARROW_BYTES);
        plan->num_slices = 2u;
        recvs[3][0] = recv(4u, 1u, 3u, g, 0u);
        plan->num_recvs = 1u;
        ops[3] = copy_op(SLICE_G, SLICE_H);
        plan->num_ops = 1u;
        l2_writes[3][0] = l2_write(window(SLICE_H, COLS, 0u, COLS), (uint32_t)output2, COLS);
        plan->num_l2_writes = 1u;
        plan->fifo.num_producers = 1u;
        break;
    }
}

static void fill_sentinel(const fifo_tile_plan_t *plan, uint32_t slice_id)
{
    const slice_desc_t *slice = get_slice((const tile_plan_t *)plan, slice_id);
    volatile uint16_t *storage =
        (volatile uint16_t *)(plan->l1_data_base + slice->l1_offset_bytes);
    for (uint32_t i = 0; i < slice->num_slots * slice->slot_bytes / ELEM; ++i)
        storage[i] = SENTINEL;
}

/* Number of planned-storage elements still holding the sentinel. */
static uint32_t count_sentinel(const fifo_tile_plan_t *plan, uint32_t slice_id)
{
    const slice_desc_t *slice = get_slice((const tile_plan_t *)plan, slice_id);
    volatile const uint16_t *storage =
        (volatile const uint16_t *)(plan->l1_data_base + slice->l1_offset_bytes);
    uint32_t count = 0u;
    for (uint32_t i = 0; i < slice->num_slots * slice->slot_bytes / ELEM; ++i)
        count += storage[i] == SENTINEL;
    return count;
}

static uint32_t check_outputs(void)
{
    uint32_t errors = 0u;
    for (uint32_t token = 0; token < NUM_TOKENS; ++token)
        for (uint32_t row = 0; row < ROWS; ++row) {
            for (uint32_t col = 0; col < COLS; ++col) {
                errors += output0[token][row][col] != input0_value(token, row, col);
                errors += output2[token][row][col] != input0_value(token, row, col);
            }
            for (uint32_t col = 0; col < WIDE_COLS; ++col)
                errors += output1[token][row][col] != input1_value(token, row, col);
        }
    return errors;
}

int main(void)
{
    const uint32_t hartid = get_hartid();

    idma_config_t idma_cfg = {.hartid = hartid};
    idma_controller_t idma = {.base = 0u, .cfg = &idma_cfg, .api = &idma_api};
    eu_config_t eu_cfg = {.hartid = hartid};
    eu_controller_t eu = {.base = 0u, .cfg = &eu_cfg, .api = &eu_api};
    fsync_config_t fsync_cfg = {.hartid = hartid};
    fsync_controller_t fsync = {.base = 0u, .cfg = &fsync_cfg, .api = &fsync_api};

    idma_init(&idma);
    eu_init(&eu);
    eu_clear_events(0xffffffffu);
    eu_idma_init(&eu, 0u);
    eu_fsync_init(&eu, 0u);
    fsync_init(&fsync);

    /* Only hart 0 clears .bss at boot; the plan tables live there, so no
     * tile may fill them before that has finished. */
    fsync_sync_global(&fsync);
    eu_fsync_wait(&eu, WFE);

    const uint32_t active = hartid < 4u;
    fifo_tile_plan_t plan;
    if (active) {
        build_plan(hartid, &plan);
        maps_fifo_init(&plan);
    }
    if (hartid == 1u)
        fill_sentinel(&plan, SLICE_S);
    if (hartid == 3u)
        fill_sentinel(&plan, SLICE_G);
    if (hartid == 0u)
        for (uint32_t token = 0; token < NUM_TOKENS; ++token)
            for (uint32_t row = 0; row < ROWS; ++row) {
                for (uint32_t col = 0; col < COLS; ++col)
                    input0[token][row][col] = input0_value(token, row, col);
                for (uint32_t col = 0; col < WIDE_COLS; ++col)
                    input1[token][row][col] = input1_value(token, row, col);
            }

    fsync_sync_global(&fsync);
    eu_fsync_wait(&eu, WFE);

    uint32_t errors = 0u;
    if (active) {
        uint32_t start = perf_get_cycles();
#ifdef FIFO_INPLACE_BLOCKING
        /* Blocking transport only, for simulator models without queued DMA. */
        for (uint32_t token = 0; token < NUM_TOKENS; ++token)
            maps_fifo_run_tile_token(&plan, token, &idma, &eu);
#else
        maps_fifo_run_tile_tokens(&plan, NUM_TOKENS, &idma, &eu);
#endif
        uint32_t cycles = perf_get_cycles() - start;

        if (hartid == 1u || hartid == 3u) {
            /* S and G are the only in-place candidates: P is strided and
             * written by two receives, and F is a send source. */
            const uint32_t slice_id = hartid == 1u ? SLICE_S : SLICE_G;
            const uint32_t untouched = count_sentinel(&plan, slice_id);
            const uint32_t elements = TOKEN_SLOTS * NARROW_BYTES / ELEM;
#if MAPS_FIFO_INPLACE_RECV
            errors += untouched != elements;
            if (hartid == 1u) {
                errors += !maps_fifo_recv_inplace_ok(&plan, 0u);
                for (uint32_t i = 1u; i < 4u; ++i)
                    errors += maps_fifo_recv_inplace_ok(&plan, i);
            }
#else
            errors += untouched != 0u;
#endif
            printf("fifo_inplace_recv tile=%u inplace=%u untouched=%u/%u cycles=%u errors=%u\n",
                   hartid, (uint32_t)MAPS_FIFO_INPLACE_RECV, untouched, elements,
                   cycles, errors);
        }
    }

    fsync_sync_global(&fsync);
    eu_fsync_wait(&eu, WFE);

    if (hartid == 0u) {
        uint32_t output_errors = check_outputs();
        errors += output_errors;
        printf("fifo_inplace_recv outputs errors=%u\n", output_errors);
    }
    return (int)errors;
}
