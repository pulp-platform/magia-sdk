#!/usr/bin/env python3
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <francesco.conti@chips.it>
"""
Generate include/test.h for ffn_qwen2.5, a Qwen2.5 FFN layer (SwiGLU MLP, no biases) with the residual:

  H = silu(X @ Wg) * (X @ Wu)       [S x I]
  Y = R + H @ Wd                    [S x E]

X is the input of the layer after the RMSNorm and R the residual stream. The golden is bit-true for the
matrix products (one RedMulE job per output element, FP16 FMA in k order) and for the SwiGLU of
kernels/swiglu_fast (FP16 Schraudolph exp and division, one FP16 rounding per instruction). It does not depend on
the mesh; the chunking of the test (FFN_CI columns of H per chunk, at least 2 chunks so that the SwiGLU of a chunk
overlaps the next one, reduction blocks FFN_KB_E and FFN_KB_I) is chosen per mesh side: the widest chunk, then
the largest blocks whose buffers fit L1.

Usage:
  python3 tests/magia/mgc/ffn_qwen2.5/gen_golden.py
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'mesh', 'mm_generator'))
from addr_map import l1_bytes, z_rows  # noqa: E402
from bittrue import matrix_multiply_with_bittrue_fma, write_fp16_array  # noqa: E402

L1_BYTES = l1_bytes() - 64  # L1_SIZE + 1, minus mgc's L1_MARGIN: the mgc buffers have all the rest of the L1

parser = argparse.ArgumentParser("Qwen2.5 FFN layer test")
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
    largest reduction blocks, whose buffers fit L1 (step 1: double-buffered X, Wg, Wu blocks and G, U chunks;
    step 2: double-buffered H, Wd blocks and resident Y block; the two steps do not share L1)."""
    assert i_size % mesh == 0, "FFN_SIZE must be a multiple of the mesh side"
    ts, ti, te = ceil_div(s_size, mesh), i_size // mesh, ceil_div(e_size, mesh)
    cis = [args.ci] if args.ci else [c for c in range(ti // 2, 0, -2) if ti % c == 0]
    for ci in cis:
        assert ti % ci == 0 and ci % 2 == 0, "FFN_CI must be even and divide FFN_SIZE / mesh"
        for kb_e in [k for k in (256, 128, 64) if e_size % k == 0]:
            for kb_i in [k for k in (256, 128, 64) if i_size % k == 0]:
                step1 = 2 * (2 * ts * kb_e + 2 * 2 * kb_e * ci + 2 * 2 * z_rows(ts) * ci)
                step2 = 2 * (2 * ts * kb_i + 2 * kb_i * te + z_rows(ts) * te)
                if step1 + step2 <= L1_BYTES:
                    return ci, kb_e, kb_i
    raise SystemExit("no FFN_CI / FFN_KB_E / FFN_KB_I fits L1 on a %dx%d mesh" % (mesh, mesh))


sched = {mesh: schedule(mesh) for mesh in args.tiles}
for mesh, (ci, kb_e, kb_i) in sched.items():
    print("tiles=%d: FFN_CI=%d (%d chunk(s)), FFN_KB_E=%d, FFN_KB_I=%d" % (mesh, ci, i_size // mesh // ci, kb_e, kb_i))

# X in [0, 1); Wg centered slightly above 0 (G ~ 0.5, with negative values for SiLU's negative branch); Wu, Wd
# positive and R in [0.5, 1.5), so that Y ~ 2 stays away from 0, where check(tol=...) is too strict
torch.manual_seed(args.seed)
X  = torch.rand(s_size, e_size).to(torch.float16).numpy()
Wg = ((torch.rand(e_size, i_size) * 4 - 1) / e_size).to(torch.float16).numpy()
Wu = (torch.rand(e_size, i_size) * (2.0 / e_size)).to(torch.float16).numpy()
Wd = (torch.rand(i_size, e_size) * (4.0 / i_size)).to(torch.float16).numpy()
R  = (torch.rand(s_size, e_size) + 0.5).to(torch.float16).numpy()



def f16(x):
    # one FP16 rounding of a result exact in float64 (products of FP16 values, fused multiply-adds)
    return np.asarray(x, dtype=np.float64).astype(np.float16)


def swiglu_fast(g, u):
    # kernels/swiglu_fast, instruction by instruction
    g, u = g.astype(np.float16), u.astype(np.float16)
    t = f16(np.maximum(g, np.float16(-11.0)).astype(np.float64) * -1477.0 + 15360.0)  # vfmax, vfmadd
    e = np.maximum(np.trunc(t.astype(np.float64)), 0).astype(np.uint16).view(np.float16)  # vfcvt.rtz.xu
    d = f16(e.astype(np.float64) + 1.0)
    p = f16(g.astype(np.float64) * u)
    a = f16(d.astype(np.float64) * 2.0**-4)
    y = (np.uint16(0x77a0) - a.view(np.uint16)).view(np.float16)  # vrsub.vx
    for _ in range(2):
        r = f16(1.0 - a.astype(np.float64) * y)  # vfnmsac
        y = f16(y.astype(np.float64) + y.astype(np.float64) * r)  # vfmacc
    y = f16(y.astype(np.float64) * 2.0**-4)
    q = f16(p.astype(np.float64) * y)
    r = f16(p.astype(np.float64) - d.astype(np.float64) * q)  # vfnmsub
    return f16(q.astype(np.float64) + y.astype(np.float64) * r)  # vfmacc


zero = np.float16(0)
print("Computing bit-true FFN..")
G = matrix_multiply_with_bittrue_fma(X, Wg, zero)
U = matrix_multiply_with_bittrue_fma(X, Wu, zero)
H = swiglu_fast(G, U)
G64, U64 = G.astype(np.float64), U.astype(np.float64)
h_ref = G64 / (1.0 + np.exp(-G64)) * U64
print("SwiGLU max abs error w.r.t. float64 (same G, U): %.3g (max |H| %.3g)" % (np.abs(H - h_ref).max(),
                                                                               np.abs(h_ref).max()))
Y = matrix_multiply_with_bittrue_fma(H, Wd, R)

X64 = X.astype(np.float64)
g = X64 @ Wg.astype(np.float64)
ref = R.astype(np.float64) + (g / (1.0 + np.exp(-g)) * (X64 @ Wu.astype(np.float64))) @ Wd.astype(np.float64)
print("Max abs error w.r.t. float64 reference: ", np.abs(Y.astype(np.float64) - ref).max())
print("Max FP16 bit distance w.r.t. float64 reference: ",
      hex(np.abs(Y.view('H').astype(np.int32) - ref.astype(np.float16).view('H').astype(np.int32)).max()))

l2_bytes = 2 * (4 * s_size * e_size + 3 * e_size * i_size + s_size * i_size)
l2ext = args.l2ext == 'yes' or (args.l2ext == 'auto' and l2_bytes > 12 * 2**20)
print("\nL2 arrays: %.1f MiB, %s" % (l2_bytes / 2**20, 'in .l2ext (needs l2_size >= 0x10000000)' if l2ext
                                    else 'in the default data region'))

os.makedirs(args.inc_dir, exist_ok=True)
with open(os.path.join(args.inc_dir, 'test.h'), 'w') as f:
    f.write('// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.\n')
    f.write('// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n')
    f.write('// SPDX-License-Identifier: Apache-2.0\n')
    f.write('// Generated by ffn_qwen2.5/gen_golden.py (S=%d, E=%d, I=%d, tiles=%s%s, seed=%d)\n'
            % (s_size, e_size, i_size, ','.join(map(str, args.tiles)), ', ci=%d' % args.ci if args.ci else '',
               args.seed))
    f.write('//\n')
    f.write('// Qwen2.5 FFN: Y[%dx%d] = R + (silu(X @ Wg) * (X @ Wu)) @ Wd, intermediate size %d\n'
            % (s_size, e_size, i_size))
    f.write('\n')
    f.write('#ifndef _TEST_FFN_QWEN25_INCLUDE_GUARD_\n')
    f.write('#define _TEST_FFN_QWEN25_INCLUDE_GUARD_\n')
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
    f.write('#error "no schedule for this MESH_X_TILES, regenerate with ffn_qwen2.5/gen_golden.py --tiles"\n')
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
    write_fp16_array(f, 'wg_inp [EMBED_SIZE * FFN_SIZE] L2EXT', Wg)
    write_fp16_array(f, 'wu_inp [EMBED_SIZE * FFN_SIZE] L2EXT', Wu)
    write_fp16_array(f, 'wd_inp [FFN_SIZE * EMBED_SIZE] L2EXT', Wd)
    write_fp16_array(f, 'y_gold [SEQ_LENGTH * EMBED_SIZE] L2EXT', Y)
    # intermediates, written by the test
    f.write('uint16_t h_l2 [SEQ_LENGTH * FFN_SIZE] L2EXT __attribute__((aligned(64))) = {0};\n')
    f.write('uint16_t y_out[SEQ_LENGTH * EMBED_SIZE] L2EXT __attribute__((aligned(64))) = {0};\n')
    f.write('\n')
    f.write('#endif //_TEST_FFN_QWEN25_INCLUDE_GUARD_\n')
