// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/*
 * ResNet18 end to end on the tile mesh, one layer at a time.
 *
 * All 49 layers of the batchnorm-folded graph (the Flatten is a no-op on NCHW and is
 * folded away by the generator) run collectively on every tile. Weights and the input
 * image sit in L2 for the whole run and each tile pulls in only the slice it needs; each
 * layer's output goes back to L2 before the next layer starts, and a mesh-wide barrier
 * between layers is what makes the next layer's reads safe.
 *
 * Which dimension gets split is left to each kernel - they all derive their shard from
 * get_hartid() internally: conv2dgemm and gemm split the output channels / GEMM rows,
 * relu and add a flat element range, maxpool2d and globalaveragepool whole channels,
 * softmax whole rows. Nothing here overrides that.
 *
 * Pass/fail is the 1000 logits, checked bit-exactly against a golden that replays the
 * kernels' own FP16 arithmetic - they are the product of 48 of the 49 layers, so this is
 * a check on the whole network. The softmax on top of them is reported but not failed
 * on, though it is reproducible now that its lane fold is vfredosum rather than the
 * unordered vfredusum. The predicted class is checked as well.
 *
 * Build RN18_MAX_LAYERS=n to stop after n layers, for bringing the thing up against the
 * per-layer tensors the generator drops in test_data/layers/.
 */

#include "tile.h"

#include "eventunit.h"
#include "fsync.h"

#include "kernel_test_utils.h"
#include "resnet18_graph.h"

#ifdef CONV2DGEMM_PROFILE
#include "performance_utils.h"
#endif

#include "add_fp16_spatz.h"
#include "conv2dgemm_fp16_spatz.h"
#include "gemm_fp16_spatz.h"
#include "globalaveragepool_fp16_spatz.h"
#include "maxpool2d_fp16_spatz.h"
#include "relu_fp16_spatz.h"
#include "softmax_fp16_spatz.h"

#define HID get_hartid()

/* The blob, loaded from the ELF into L2 by resnet18_data.S. */
extern const uint8_t resnet18_data[];

/*
 * Every layer's output, all resident at once, so a residual branch can read a tensor
 * produced several layers earlier without any liveness analysis. NOLOAD, so it costs
 * nothing in the ELF; nothing zeroes it, which is fine because every element is written
 * before it is read.
 */
static float16 resnet18_arena[RN18_ARENA_BYTES / sizeof(float16)]
    __attribute__((section(".l2_arena"), aligned(4)));

extern uint32_t _spatz_binary_start; /* CV32 linker script */

static inline float16 *arena_at(uint32_t off)
{
    return (float16 *)((uintptr_t)resnet18_arena + off);
}

static inline const float16 *blob_at(uint32_t off)
{
    return (const float16 *)((uintptr_t)resnet18_data + off);
}

static inline const float16 *operand(uint8_t space, uint32_t off)
{
    return space == RN18_SPACE_BLOB ? blob_at(off) : arena_at(off);
}

static void run_layer(const rn18_layer_t *L)
{
    const float16 *src0 = operand(L->src0_space, L->src0_off);
    float16 *dst        = arena_at(L->dst_off);

    switch (L->op) {
    case RN18_CONV: {
        uint32_t in_shape[4]  = {L->in_shape[0], L->in_shape[1], L->in_shape[2], L->in_shape[3]};
        uint32_t out_shape[4] = {
            L->out_shape[0], L->out_shape[1], L->out_shape[2], L->out_shape[3]};

        MAGIA_conv2dgemm_fp16_spatz(src0,
                                    blob_at(L->w_off),
                                    L->has_bias ? blob_at(L->b_off) : NULL,
                                    dst,
                                    in_shape,
                                    out_shape,
                                    L->kernel_h,
                                    L->kernel_w,
                                    L->stride_h,
                                    L->stride_w,
                                    L->pad_h,
                                    L->pad_w,
                                    1, /* group */
                                    (int)L->has_bias);
        break;
    }

    case RN18_RELU:
        MAGIA_relu_fp16_spatz(src0, dst, L->numel);
        break;

    case RN18_ADD:
        MAGIA_add_fp16_spatz(src0, operand(L->src1_space, L->src1_off), dst, L->numel);
        break;

    case RN18_MAXPOOL: {
        uint32_t in_shape[4]  = {L->in_shape[0], L->in_shape[1], L->in_shape[2], L->in_shape[3]};
        uint32_t out_shape[4] = {
            L->out_shape[0], L->out_shape[1], L->out_shape[2], L->out_shape[3]};

        MAGIA_maxpool2d_fp16_spatz(src0,
                                   dst,
                                   L->kernel_h,
                                   L->kernel_w,
                                   L->stride_h,
                                   L->stride_w,
                                   L->pad_h,
                                   L->pad_w,
                                   in_shape,
                                   out_shape);
        break;
    }

    case RN18_GLOBALAVERAGEPOOL: {
        uint32_t in_shape[4] = {L->in_shape[0], L->in_shape[1], L->in_shape[2], L->in_shape[3]};

        MAGIA_globalaveragepool_fp16_spatz(src0, dst, in_shape);
        break;
    }

    case RN18_GEMM: {
        /*
         * Called transposed: A is the [1000, 512] fc weight, B the 512-long embedding as
         * a column, Y a [1000, 1] column - which is the same bytes as [1, 1000]. The
         * kernel shards the GEMM's M, so the ONNX orientation (M = 1) would leave the
         * whole layer on tile 0. N = 1 makes the row stride 2 bytes, so the kernel's
         * misalignment guard takes its scalar path; at 512 MACs per output row that
         * costs nothing.
         */
        uint32_t a_shape[2] = {L->out_shape[1], L->in_shape[1]};
        uint32_t b_shape[2] = {L->in_shape[1], 1};
        uint32_t c_shape[2] = {L->out_shape[1], 1};
        uint32_t y_shape[2] = {L->out_shape[1], 1};

        MAGIA_gemm_fp16_spatz(blob_at(L->w_off),
                              src0,
                              blob_at(L->b_off),
                              (float16)1.0f,
                              (float16)1.0f,
                              0,
                              0,
                              a_shape,
                              b_shape,
                              c_shape,
                              y_shape,
                              dst);
        break;
    }

    case RN18_SOFTMAX: {
        /* One row of numel: the kernel takes rows = shape[0]*shape[1]*shape[2]. */
        uint32_t in_shape[4] = {1, 1, 1, L->numel};

        MAGIA_softmax_fp16_spatz(src0, dst, in_shape);
        break;
    }

    default:
        if (HID == 0)
            printf("[CV32 (%d)] unhandled op %d in layer %s\n", HID, L->op, L->name);
        break;
    }
}

int main(void)
{
    fsync_config_t fsync_cfg;
    fsync_controller_t fsync_ctrl;
    eu_config_t eu_cfg;
    eu_controller_t eu_ctrl;
    bool ok = true;

    if (HID == 0)
        printf("\n########## ONNX_RESNET18 (224x224, %d layers) on %d Tiles ##########\n\n",
               RN18_MAX_LAYERS,
               NUM_HARTS);

    /* Brings up the event unit and Spatz. Must come before eu_fsync_init: it calls
     * eu_init, which clears the event mask. */
    kt_spatz_init((uint32_t)&_spatz_binary_start);

    eu_cfg.hartid = HID;
    eu_ctrl.base  = 0;
    eu_ctrl.cfg   = &eu_cfg;
    eu_ctrl.api   = &eu_api;
    eu_fsync_init(&eu_ctrl, 0);

    fsync_cfg.hartid = HID;
    fsync_ctrl.base  = 0;
    fsync_ctrl.cfg   = &fsync_cfg;
    fsync_ctrl.api   = &fsync_api;
    fsync_init(&fsync_ctrl);

    /* crt0 clears .bss from hart 0 only, so no tile may touch L2 until that is done. */
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WFE);

    for (uint32_t i = 0; i < RN18_MAX_LAYERS; i++) {
        const rn18_layer_t *L = &resnet18_layers[i];

        if (HID == 0)
            printf("[CV32 (%d)] layer %s\n", HID, L->name);

#ifdef CONV2DGEMM_PROFILE
        unsigned int t_layer = perf_get_cycles();
#endif

        run_layer(L);

#ifdef CONV2DGEMM_PROFILE
        unsigned int t_barrier = perf_get_cycles();
#endif

        /* Every tile's shard of this layer is in L2 before any tile reads it back. */
        fsync_sync_global(&fsync_ctrl);
        eu_fsync_wait(&eu_ctrl, WFE);

#ifdef CONV2DGEMM_PROFILE
        /* Barrier time is mostly this tile waiting for the slowest one, so it doubles as
         * a load-imbalance readout. Printed from every tile, not just hart 0. */
        printf("[CV32 (%d)] [prof] %s compute %u barrier %u\n",
               HID,
               L->name,
               t_barrier - t_layer,
               perf_get_cycles() - t_barrier);
#endif
    }

    spatz_clk_dis();

    /*
     * Partial run: there is no compiled-in golden for an intermediate layer, so dump the
     * head and tail of the last one computed. test_data/layers/<nn>_<op>.npy holds the
     * same tensor, so a few values are enough to tell whether that layer is right.
     */
    if (HID == 0 && RN18_MAX_LAYERS != RN18_NUM_LAYERS) {
        const rn18_layer_t *L = &resnet18_layers[RN18_MAX_LAYERS - 1];
        const float16 *y      = arena_at(L->dst_off);

        printf("\n[CV32 (%d)] stopped after %s (%d elements)\n", HID, L->name, L->numel);
        printf("[CV32 (%d)] head:", HID);
        for (uint32_t i = 0; i < 8 && i < L->numel; i++)
            printf(" %04x", kt_bits(y[i]));
        printf("\n[CV32 (%d)] tail:", HID);
        for (uint32_t i = (L->numel > 8 ? L->numel - 8 : 0); i < L->numel; i++)
            printf(" %04x", kt_bits(y[i]));
        printf("\n\n");
    }

    if (HID == 0 && RN18_MAX_LAYERS == RN18_NUM_LAYERS) {
        const float16 *logits        = arena_at(RN18_LOGITS_OFF);
        const float16 *golden_logits = blob_at(RN18_GOLDEN_LOGITS_OFF);
        const float16 *output        = arena_at(RN18_OUTPUT_OFF);
        const float16 *golden_output = blob_at(RN18_GOLDEN_OUTPUT_OFF);
        uint32_t top1                = 0;

        printf("\n[CV32 (%d)] logits (bit-exact, pass/fail):\n", HID);
        ok = kt_check(logits, golden_logits, RN18_LOGITS_LEN, RN18_LOGITS_ULP_TOLL);

        for (uint32_t i = 1; i < RN18_OUTPUT_LEN; i++)
            if (output[i] > output[top1])
                top1 = i;

        /* Reported, not failed on. See the note in the generated header. */
        printf("[CV32 (%d)] softmax (informational):\n", HID);
        (void)kt_check(output, golden_output, RN18_OUTPUT_LEN, 0);

        printf("[CV32 (%d)] top-1 class %d, expected %d\n", HID, top1, RN18_TOP1);
        if (top1 != RN18_TOP1)
            ok = false;

        printf("[CV32 (%d)] Test %s\n", HID, ok ? "SUCCESS" : "FAILED");
        printf("\n####################################################################\n\n");
    }

    return ok ? 0 : -1;
}
