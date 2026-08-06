#!/usr/bin/env python3
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Generator for the nn_is_bench embedded dataset.
#
# Emits two headers next to each other:
#   nn_is_size.h  -- geometry only (included by nn_is_params.h, and therefore by
#                    the Spatz and PULP task sources: kept tiny on purpose)
#   nn_is_data.h  -- the fp16 matrices themselves, included only by main.c
#
# The matrices land in .data (not .rodata/.bss) so that the ELF image carries
# the full working set: that is what the VFIO host has to push into L2 on every
# single iteration, which is exactly the overhead we want to measure.
#
# Usage:
#   python3 scripts/gen_nn_is_data.py --size 128 \
#       -o tests/spatz_pulp_on_magia/nn_is_bench/include
#   python3 scripts/gen_nn_is_data.py --m 512 --n 512 --k 512 -o <dir>

import argparse
import os
import sys

import numpy as np

HEADER = """// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// GENERATED FILE -- do not edit.
// Regenerate with:
//   python3 scripts/gen_nn_is_data.py {cmd}
"""


def fp16_bits(a):
    """View a float16 array as its raw 16-bit patterns."""
    return a.astype(np.float16).view(np.uint16)


def emit_u16_array(fh, name, values, per_line=12):
    fh.write("uint16_t %s[%d] = {\n" % (name, values.size))
    flat = values.reshape(-1)
    for i in range(0, flat.size, per_line):
        chunk = flat[i:i + per_line]
        fh.write("    " + " ".join("0x%04x," % v for v in chunk) + "\n")
    fh.write("};\n\n")


def emit_u32_array(fh, name, values, per_line=8):
    fh.write("const uint32_t %s[%d] = {\n" % (name, values.size))
    flat = values.reshape(-1)
    for i in range(0, flat.size, per_line):
        chunk = flat[i:i + per_line]
        fh.write("    " + " ".join("%du," % v for v in chunk) + "\n")
    fh.write("};\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=None,
                    help="shorthand for --m/--n/--k all equal")
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--timeslots", type=int, default=4)
    ap.add_argument("--seed", type=lambda v: int(v, 0), default=0xC0FFEE)
    ap.add_argument("--samples-per-tile", type=int, default=32,
                    help="golden samples embedded per mesh tile block")
    ap.add_argument("--mesh", type=int, default=4,
                    help="mesh side used to lay out the golden samples")
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()

    if args.size is not None:
        args.m = args.n = args.k = args.size

    m, n, k, ts, mesh = args.m, args.n, args.k, args.timeslots, args.mesh

    # Geometry sanity: these are the divisibility rules the C code relies on.
    if m % mesh or n % mesh or k % mesh:
        sys.exit("M/N/K must all be divisible by the mesh side (%d)" % mesh)
    if k % ts:
        sys.exit("K must be divisible by --timeslots")
    if (m // mesh) % 8:
        sys.exit("M/mesh (rows per tile) must be a multiple of 8 "
                 "(one row slice per PULP core)")

    rng = np.random.default_rng(args.seed)

    # Values in +/-[0.25, 2.0): the GEMM accumulates N of their products, so the
    # magnitude stays far below the fp16 range even for N = 512.
    def rand_block(shape):
        mag = rng.uniform(0.25, 2.0, size=shape)
        sign = rng.integers(0, 2, size=shape) * 2 - 1
        return (mag * sign).astype(np.float16)

    x = rand_block((m, n))
    w = rand_block((n, k))

    # Bias broadcast along the rows; deliberately non-zero so the array cannot
    # be collapsed into .bss by the compiler.
    bias_row = (rng.uniform(-0.5, 0.5, size=k)).astype(np.float16)
    y = np.broadcast_to(bias_row, (m, k)).astype(np.float16).copy()

    # Golden: accumulate in fp32 (RedMulE keeps a wider internal accumulator),
    # then round the result to fp16 as the accelerator writes it back.
    z = (x.astype(np.float32) @ w.astype(np.float32) +
         y.astype(np.float32)).astype(np.float16)

    # Golden samples, spread over each mesh tile's post-GEMM block so every tile
    # has something local to check.
    th, pb = m // mesh, k // mesh
    idx = []
    for ty in range(mesh):
        for tx in range(mesh):
            rows = rng.integers(0, th, size=args.samples_per_tile) + ty * th
            cols = rng.integers(0, pb, size=args.samples_per_tile) + tx * pb
            idx.extend((rows * k + cols).tolist())
    idx = np.array(sorted(set(idx)), dtype=np.uint32)
    val = fp16_bits(z.reshape(-1)[idx])

    # Per-sample absolute tolerance, in millis.
    #
    # A relative tolerance on the *result* is meaningless here: each output is a
    # sum of N products of magnitude ~1 with random signs, so cancellation can
    # leave a result far smaller than the partial sums that produced it, and the
    # relative error is then unbounded. The right bound is on the accumulation:
    # with unit roundoff eps and N rounding steps whose errors add in quadrature,
    #     |err| ~ eps * sqrt(N) * sum_n |x[m,n] * w[n,k]|
    # eps = 2^-11 for fp16 (10 mantissa bits plus the implicit one).
    eps      = 2.0 ** -11
    sum_abs  = np.abs(x.astype(np.float32)) @ np.abs(w.astype(np.float32))
    tol_abs  = eps * np.sqrt(float(n)) * sum_abs.reshape(-1)[idx]
    # Floor at 20 millis so near-zero rows still get a usable window.
    tol = np.maximum(np.rint(tol_abs * 1000.0), 20.0).astype(np.uint32)

    os.makedirs(args.outdir, exist_ok=True)
    cmd = " ".join(sys.argv[1:])

    size_path = os.path.join(args.outdir, "nn_is_size.h")
    with open(size_path, "w") as fh:
        fh.write(HEADER.format(cmd=cmd))
        fh.write("""
#ifndef NN_IS_SIZE_H_
#define NN_IS_SIZE_H_

#define NN_M          ({m})
#define NN_N          ({n})
#define NN_K          ({k})
#define NN_TIMESLOTS  ({ts})
#define NN_NSAMPLES   ({ns})
#define NN_DATA_SEED  (0x{seed:x}u)

#endif /* NN_IS_SIZE_H_ */
""".format(m=m, n=n, k=k, ts=ts, ns=idx.size, seed=args.seed))

    data_path = os.path.join(args.outdir, "nn_is_data.h")
    with open(data_path, "w") as fh:
        fh.write(HEADER.format(cmd=cmd))
        fh.write("""
#ifndef NN_IS_DATA_H_
#define NN_IS_DATA_H_

#include <stdint.h>
#include "nn_is_size.h"

/* All four working buffers live in .data so the ELF image carries them: the
 * VFIO host re-pushes the whole image on every iteration by design. */

""")
        emit_u16_array(fh, "nn_x_inp", fp16_bits(x))
        emit_u16_array(fh, "nn_w_inp", fp16_bits(w))
        emit_u16_array(fh, "nn_y_inp", fp16_bits(y))
        fh.write("/* int16 Q0.15 softmax output, written by the PULP cluster. */\n")
        fh.write("uint16_t nn_q_out[%d] = {1};\n\n" % (m * k))
        fh.write("/* Golden samples of Z = X*W + Y_bias: flat index, fp16 bit\n"
                 " * pattern, and the fp16 accumulation error bound in millis. */\n")
        emit_u32_array(fh, "nn_z_idx", idx)
        fh.write("const ")
        emit_u16_array(fh, "nn_z_val", val)
        emit_u32_array(fh, "nn_z_tol", tol)
        fh.write("#endif /* NN_IS_DATA_H_ */\n")

    in_kib = (x.size + w.size) * 2 / 1024.0
    print("wrote %s" % size_path)
    print("wrote %s" % data_path)
    print("  M=%d N=%d K=%d timeslots=%d mesh=%dx%d" % (m, n, k, ts, mesh, mesh))
    print("  per-tile: rows=%d inner=%d k-slice=%d post-block=%dx%d"
          % (th, n // mesh, k // ts, th, pb))
    print("  host payload (X+W) = %.0f KiB, golden samples = %d" % (in_kib, idx.size))
    print("  golden tolerance (millis): min=%d median=%d max=%d"
          % (tol.min(), int(np.median(tol)), tol.max()))


if __name__ == "__main__":
    main()
