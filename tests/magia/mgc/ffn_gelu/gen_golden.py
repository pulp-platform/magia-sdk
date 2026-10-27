#!/usr/bin/env python3
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <francesco.conti@chips.it>
"""
Generate include/test.h for ffn_gelu, a ViT FFN layer (GELU MLP, no biases) with the residual:

  H = gelu(X @ W1)                  [S x I]
  Y = R + H @ W2                    [S x E]

X is the input of the layer after the LayerNorm and R the residual stream. The golden is bit-true for the
matrix products (one RedMulE job per output element, FP16 FMA in k order) and for the GELU of
kernels/gelu_fast (tanh form, FP16 Schraudolph exp and division, one FP16 rounding per instruction); the
error w.r.t. the exact (erf) GELU in float64 is printed. It does not depend on
the mesh; the chunking of the test (FFN_CI columns of H per chunk, at least 2 chunks so that the GELU of a chunk
overlaps the next one, reduction blocks FFN_KB_E and FFN_KB_I) is chosen per mesh side: the widest chunk, then
the largest blocks whose buffers fit L1.

Usage:
  python3 tests/magia/mgc/ffn_gelu/gen_golden.py
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'mesh', 'mm_generator'))
from bittrue import matrix_multiply_with_bittrue_fma, write_fp16_array  # noqa: E402

L1_BYTES = 0xE0000  # L1_SIZE + 1: the mgc buffers have all the L1 (stack and .tile_bss are below L1_BASE)

parser = argparse.ArgumentParser("ViT FFN layer test")
parser.add_argument('--s_size', type=int, default=512)   # SEQ_LENGTH
parser.add_argument('--e_size', type=int, default=128)   # EMBED_SIZE
parser.add_argument('--i_size', type=int, default=512)   # FFN_SIZE (intermediate)
parser.add_argument('--tiles', type=int, nargs='+', default=[2, 4, 8, 16])
parser.add_argument('--ci', type=int, default=None)      # force FFN_CI (columns of H per chunk)
parser.add_argument('--seed', type=int, default=42)
# place the L2 arrays in the .l2ext section (above the default gvsoc L2: run with l2_size >= 0x10000000), by
# default only if they do not fit the 15 MiB of the default data region (with some room for the rest)
parser.add_argument('--l2ext', choices=['auto', 'yes', 'no'], default='auto')
parser.add_argument('--inc_dir', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'include'))
args = parser.parse_args()

s_size, e_size, i_size = args.s_size, args.e_size, args.i_size


def ceil_div(a, b):
    return (a + b - 1) // b


def schedule(mesh):
    """(FFN_CI, FFN_KB_E, FFN_KB_I) for a mesh side: the widest chunk of H with at least 2 chunks, then the
    largest reduction blocks, whose buffers fit L1 (step 1: double-buffered X, W1 blocks and H chunks;
    step 2: double-buffered H, W2 blocks and resident Y block; the two steps do not share L1)."""
    assert i_size % mesh == 0, "FFN_SIZE must be a multiple of the mesh side"
    ts, ti, te = ceil_div(s_size, mesh), i_size // mesh, ceil_div(e_size, mesh)
    cis = [args.ci] if args.ci else [c for c in range(ti // 2, 0, -2) if ti % c == 0]
    for ci in cis:
        assert ti % ci == 0 and ci % 2 == 0, "FFN_CI must be even and divide FFN_SIZE / mesh"
        for kb_e in [k for k in (256, 128, 64) if e_size % k == 0]:
            for kb_i in [k for k in (256, 128, 64) if i_size % k == 0]:
                step1 = 2 * (2 * ts * kb_e + 2 * kb_e * ci + 2 * ts * ci)
                step2 = 2 * (2 * ts * kb_i + 2 * kb_i * te + ts * te)
                if step1 + step2 <= L1_BYTES:
                    return ci, kb_e, kb_i
    raise SystemExit("no FFN_CI / FFN_KB_E / FFN_KB_I fits L1 on a %dx%d mesh" % (mesh, mesh))


sched = {mesh: schedule(mesh) for mesh in args.tiles}
for mesh, (ci, kb_e, kb_i) in sched.items():
    print("tiles=%d: FFN_CI=%d (%d chunk(s)), FFN_KB_E=%d, FFN_KB_I=%d" % (mesh, ci, i_size // mesh // ci, kb_e, kb_i))

# X in [0, 1); W1 zero-mean, X @ W1 with std ~2 (both branches of GELU, and its saturation); W2 positive and
# R in [0.5, 1.5), so that Y stays away from 0, where check(tol=...) is too strict
torch.manual_seed(args.seed)
X  = torch.rand(s_size, e_size).to(torch.float16).numpy()
W1 = ((torch.rand(e_size, i_size) * 2 - 1) * (6.0 / np.sqrt(e_size))).to(torch.float16).numpy()
W2 = (torch.rand(i_size, e_size) * (2.0 / i_size)).to(torch.float16).numpy()
R  = (torch.rand(s_size, e_size) + 0.5).to(torch.float16).numpy()



def f16(x):
    # one FP16 rounding of a result exact in float64 (products of FP16 values, fused multiply-adds)
    return np.asarray(x, dtype=np.float64).astype(np.float16)


def gelu_fast(x):
    # kernels/gelu_fast, instruction by instruction
    x = x.astype(np.float16)
    x64 = x.astype(np.float64)
    t = f16(x64 * x64)
    t = f16(t.astype(np.float64) * float(np.float16(0.044715)) + 1.0)  # vfmadd
    z = f16(t.astype(np.float64) * x64)
    z = np.minimum(np.maximum(z, np.float16(-6.875)), np.float16(16.0))  # vfmax, vfmin
    t = f16(z.astype(np.float64) * -2356.0 + 15360.0)  # vfmadd
    e = np.maximum(np.trunc(t.astype(np.float64)), 0).astype(np.uint16).view(np.float16)  # vfcvt.rtz.xu
    d = f16(e.astype(np.float64) + 1.0)
    a = f16(d.astype(np.float64) * 2.0**-4)
    y = (np.uint16(0x77a0) - a.view(np.uint16)).view(np.float16)  # vrsub.vx
    for _ in range(2):
        r = f16(1.0 - a.astype(np.float64) * y)  # vfnmsac
        y = f16(y.astype(np.float64) + y.astype(np.float64) * r)  # vfmacc
    y = f16(y.astype(np.float64) * 2.0**-4)
    q = f16(x64 * y)
    r = f16(x64 - d.astype(np.float64) * q)  # vfnmsub
    return f16(q.astype(np.float64) + y.astype(np.float64) * r)  # vfmacc


def gelu_ref(x):
    # exact GELU, x * Phi(x), in float64
    return 0.5 * x * (1.0 + torch.erf(torch.from_numpy(x) / np.sqrt(2.0)).numpy())


zero = np.float16(0)
print("Computing bit-true FFN..")
U = matrix_multiply_with_bittrue_fma(X, W1, zero)
H = gelu_fast(U)
h_ref = gelu_ref(U.astype(np.float64))
print("GELU max abs error w.r.t. exact float64 GELU (same X @ W1): %.3g (X @ W1 in [%.3g, %.3g])"
      % (np.abs(H - h_ref).max(), U.min(), U.max()))
Y = matrix_multiply_with_bittrue_fma(H, W2, R)

X64 = X.astype(np.float64)
ref = R.astype(np.float64) + gelu_ref(X64 @ W1.astype(np.float64)) @ W2.astype(np.float64)
print("Max abs error w.r.t. float64 reference: ", np.abs(Y.astype(np.float64) - ref).max())
print("Max FP16 bit distance w.r.t. float64 reference: ",
      hex(np.abs(Y.view('H').astype(np.int32) - ref.astype(np.float16).view('H').astype(np.int32)).max()))

l2_bytes = 2 * (4 * s_size * e_size + 2 * e_size * i_size + s_size * i_size)
l2ext = args.l2ext == 'yes' or (args.l2ext == 'auto' and l2_bytes > 12 * 2**20)
print("\nL2 arrays: %.1f MiB, %s" % (l2_bytes / 2**20, 'in .l2ext (needs l2_size >= 0x10000000)' if l2ext
                                    else 'in the default data region'))

os.makedirs(args.inc_dir, exist_ok=True)
with open(os.path.join(args.inc_dir, 'test.h'), 'w') as f:
    f.write('// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.\n')
    f.write('// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n')
    f.write('// SPDX-License-Identifier: Apache-2.0\n')
    f.write('// Generated by ffn_gelu/gen_golden.py (S=%d, E=%d, I=%d, tiles=%s%s, seed=%d)\n'
            % (s_size, e_size, i_size, ','.join(map(str, args.tiles)), ', ci=%d' % args.ci if args.ci else '',
               args.seed))
    f.write('//\n')
    f.write('// ViT FFN: Y[%dx%d] = R + gelu(X @ W1) @ W2, intermediate size %d\n'
            % (s_size, e_size, i_size))
    f.write('\n')
    f.write('#ifndef _TEST_FFN_GELU_INCLUDE_GUARD_\n')
    f.write('#define _TEST_FFN_GELU_INCLUDE_GUARD_\n')
    f.write('\n')
    f.write('#define SEQ_LENGTH (%d)\n' % s_size)
    f.write('#define EMBED_SIZE (%d)\n' % e_size)
    f.write('#define FFN_SIZE   (%d)\n' % i_size)
    f.write('\n')
    # schedule per mesh side, selected by MESH_X_TILES
    f.write('#include "addr_map/tile_config.h"\n')
    f.write('\n')
    for i, mesh in enumerate(args.tiles):
        f.write('#%s MESH_X_TILES == %d\n' % ('if' if i == 0 else 'elif', mesh))
        f.write('#define FFN_CI   (%d)\n' % sched[mesh][0])
        f.write('#define FFN_KB_E (%d)\n' % sched[mesh][1])
        f.write('#define FFN_KB_I (%d)\n' % sched[mesh][2])
    f.write('#else\n')
    f.write('#error "no schedule for this MESH_X_TILES, regenerate with ffn_gelu/gen_golden.py --tiles"\n')
    f.write('#endif\n')
    f.write('\n')
    if l2ext:
        f.write('// %.1f MiB of L2 arrays: in .l2ext, above the default L2 (build and run gvsoc with\n' % (l2_bytes / 2**20))
        f.write('// l2_size >= 0x10000000)\n')
        f.write('#define L2EXT __attribute__((section(".l2ext")))\n')
    else:
        f.write('#define L2EXT\n')
    f.write('\n')
    write_fp16_array(f, 'x_inp  [SEQ_LENGTH * EMBED_SIZE] L2EXT', X)
    write_fp16_array(f, 'res_inp[SEQ_LENGTH * EMBED_SIZE] L2EXT', R)
    write_fp16_array(f, 'w1_inp [EMBED_SIZE * FFN_SIZE] L2EXT', W1)
    write_fp16_array(f, 'w2_inp [FFN_SIZE * EMBED_SIZE] L2EXT', W2)
    write_fp16_array(f, 'y_gold [SEQ_LENGTH * EMBED_SIZE] L2EXT', Y)
    # intermediates, written by the test
    f.write('uint16_t h_l2 [SEQ_LENGTH * FFN_SIZE] L2EXT __attribute__((aligned(64))) = {0};\n')
    f.write('uint16_t y_out[SEQ_LENGTH * EMBED_SIZE] L2EXT __attribute__((aligned(64))) = {0};\n')
    f.write('\n')
    f.write('#endif //_TEST_FFN_GELU_INCLUDE_GUARD_\n')
