#!/usr/bin/env python3
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <francesco.conti@chips.it>
"""
Generate include/test.h for mhsa_pipeopt4, multi-head self-attention:

  Q = X @ Wq, Kt = Wk^T @ X^T, V = X @ Wv                      [S x H*P], [H*P x S], [S x H*P]
  O[:, h*P:(h+1)*P] = softmax(Q_h @ Kt_h / sqrt(P)) @ V_h      for every head h
  Y = O @ Wo                                                   [S x E]

X is given both as x_inp [S x E] and transposed as xt_inp [E x S], and Wk transposed as
wkt_inp [H*P x E], so that RedMulE computes Kt directly. The golden is bit-true: the projections
are one RedMulE job per output element (FP16 FMA in k order), each head is the FlatAttention
model of mm_generator/flat_attn.py (online softmax over K/V blocks split over the mesh columns,
Schraudolph exp, binary-tree reduction of O), so there is one golden per mesh side.

Usage:
  python3 tests/magia/mgc/mhsa_pipeopt4/gen_golden.py
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'mesh', 'mm_generator'))
from bittrue import matrix_multiply_with_bittrue_fma, write_fp16_array  # noqa: E402

parser = argparse.ArgumentParser("Multi-head self-attention test")
parser.add_argument('--s_size', type=int, default=1024)   # SEQ_LENGTH
parser.add_argument('--e_size', type=int, default=128)    # EMBED_SIZE
parser.add_argument('--p_size', type=int, default=64)     # PROJ_SIZE (head size)
parser.add_argument('--n_heads', type=int, default=2)     # NUM_HEADS
parser.add_argument('--block_kv', type=int, default=512)  # S_BLOCK_SIZE_KV in the .mgc
parser.add_argument('--tiles', type=int, nargs='+', default=[2, 4, 8, 16])
parser.add_argument('--q_std', type=float, default=1.15)  # std of Q and K, as U(-2, 2) in flat_attn.py
parser.add_argument('--seed', type=int, default=42)
# place the L2 arrays in the .l2ext section (above the default gvsoc L2: run with l2_size >= 0x10000000), by
# default only if they do not fit the 15 MiB of the default data region (with some room for the rest)
parser.add_argument('--l2ext', choices=['auto', 'yes', 'no'], default='auto')
parser.add_argument('--inc_dir', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'include'))
args = parser.parse_args()

s_size, e_size, p_size, n_heads, block = args.s_size, args.e_size, args.p_size, args.n_heads, args.block_kv
hp_size = n_heads * p_size
assert s_size % block == 0
for mesh_x in args.tiles:
    assert block % mesh_x == 0
    assert mesh_x & (mesh_x - 1) == 0, "binary-tree reduction of O assumes a power-of-2 mesh"

# X in [0, 1); Wq, Wk zero-mean, scaled so that Q and K have std ~q_std (peaked softmax);
# Wv, Wo positive, so that V ~ 0.5 and Y ~ 1 stay away from 0, where check(tol=...) is too strict
torch.manual_seed(args.seed)
a = 3 * args.q_std / np.sqrt(e_size)  # var(Q) = E * E[X^2] * a^2 / 3 = E * a^2 / 9
X  = torch.rand(s_size, e_size).to(torch.float16).numpy()
Wq = ((torch.rand(e_size, hp_size) * 2 - 1) * a).to(torch.float16).numpy()
Wk = ((torch.rand(e_size, hp_size) * 2 - 1) * a).to(torch.float16).numpy()
Wv = (torch.rand(e_size, hp_size) * (2.0 / e_size)).to(torch.float16).numpy()
Wo = (torch.rand(hp_size, e_size) * (4.0 / hp_size)).to(torch.float16).numpy()
Xt, Wkt = np.ascontiguousarray(X.T), np.ascontiguousarray(Wk.T)


def f16(x):
    # every Spatz op computes in FP16: exact in float64, then one rounding
    return np.asarray(x, dtype=np.float64).astype(np.float16)


def fast_exp(x):
    # Schraudolph exp of the expminus Spatz kernels: clamp, x*COEF+BIAS, rtz to the FP16 bits
    x = np.clip(x, np.float16(-10.0), np.float16(10.0))
    y = f16(f16(x.astype(np.float64) * 1477.0).astype(np.float64) + 15360.0)
    return np.trunc(y.astype(np.float64)).astype(np.uint16).view(np.float16)


def rowsum(x):
    # sequential FP16 accumulation along the last axis
    acc = x[..., 0]
    for c in range(1, x.shape[-1]):
        acc = f16(acc.astype(np.float64) + x[..., c])
    return acc


scale = np.float16(1.0 / np.sqrt(np.float32(p_size)))
zero = np.float16(0)


def flat_attn(Q, Kt, V, mesh_x):
    # as flat_attn() in mm_generator/flat_attn.py, for one head
    tile_kv = block // mesh_x
    o = [np.zeros((s_size, p_size), dtype=np.float16) for _ in range(mesh_x)]
    for sikv in range(s_size // block):
        cols = [slice(sikv * block + jj * tile_kv, sikv * block + (jj + 1) * tile_kv) for jj in range(mesh_x)]

        attn = [matrix_multiply_with_bittrue_fma(Q, Kt[:, c], zero) for c in cols]
        attn = [f16(a.astype(np.float64) * scale) for a in attn]

        m_tile = [a.max(axis=1) for a in attn]
        if sikv > 0:
            m_tile = [np.maximum(m, rowm) for m in m_tile]
        rowm = np.max(m_tile, axis=0)

        attn = [fast_exp(f16(a.astype(np.float64) - rowm[:, None])) for a in attn]
        if sikv > 0:
            alpha = fast_exp(f16(rowmprev.astype(np.float64) - rowm))

        rows = rowsum(np.stack([rowsum(a) for a in attn], axis=-1))
        if sikv > 0:
            rows = f16(f16(alpha.astype(np.float64) * rowsprev).astype(np.float64) + rows)
            o = [f16(oj.astype(np.float64) * alpha[:, None]) for oj in o]

        o = [matrix_multiply_with_bittrue_fma(a, V[c, :], oj) for a, c, oj in zip(attn, cols, o)]

        rowmprev, rowsprev = rowm, rows

    o = [f16(oj.astype(np.float64) / rows[:, None]) for oj in o]
    stride = 1
    while stride < mesh_x:
        for jj in range(0, mesh_x, 2 * stride):
            o[jj] = f16(o[jj].astype(np.float64) + o[jj + stride])
        stride *= 2
    return o[0]


print("Computing bit-true Q/K/V projections..")
Q  = matrix_multiply_with_bittrue_fma(X, Wq, zero)
Kt = matrix_multiply_with_bittrue_fma(Wkt, Xt, zero)
V  = matrix_multiply_with_bittrue_fma(X, Wv, zero)
heads = [slice(h * p_size, (h + 1) * p_size) for h in range(n_heads)]

# float64 reference with the exact exp, for information only
X64 = X.astype(np.float64)
ref = []
for hs in heads:
    a = (X64 @ Wq[:, hs]) @ (X64 @ Wk[:, hs]).T * scale
    a = np.exp(a - a.max(axis=1, keepdims=True))
    ref.append((a / a.sum(axis=1, keepdims=True)) @ (X64 @ Wv[:, hs]))
ref = np.concatenate(ref, axis=1) @ Wo.astype(np.float64)

Y = {}
for mesh_x in args.tiles:
    print("\nComputing bit-true MHSA, tiles=%d.." % mesh_x)
    O = np.concatenate([flat_attn(Q[:, hs], Kt[hs, :], V[:, hs], mesh_x) for hs in heads], axis=1)
    Y[mesh_x] = matrix_multiply_with_bittrue_fma(O, Wo, zero)
    print("Max abs error w.r.t. float64 reference: ", np.abs(Y[mesh_x].astype(np.float64) - ref).max())
    print("Max FP16 bit distance w.r.t. float64 reference: ",
          hex(np.abs(Y[mesh_x].view('H').astype(np.int32) - f16(ref).view('H').astype(np.int32)).max()))

l2_bytes = 2 * (4 * s_size * e_size + 4 * e_size * hp_size + 4 * s_size * hp_size)
l2ext = args.l2ext == 'yes' or (args.l2ext == 'auto' and l2_bytes > 12 * 2**20)
print("\nL2 arrays: %.1f MiB, %s" % (l2_bytes / 2**20, 'in .l2ext (needs l2_size >= 0x10000000)' if l2ext
                                    else 'in the default data region'))

os.makedirs(args.inc_dir, exist_ok=True)
with open(os.path.join(args.inc_dir, 'test.h'), 'w') as f:
    f.write('// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.\n')
    f.write('// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n')
    f.write('// SPDX-License-Identifier: Apache-2.0\n')
    f.write('// Generated by mhsa_pipeopt4/gen_golden.py (S=%d, E=%d, P=%d, H=%d, block_kv=%d, tiles=%s, '
            'q_std=%g, seed=%d)\n' % (s_size, e_size, p_size, n_heads, block, ','.join(map(str, args.tiles)),
                                      args.q_std, args.seed))
    f.write('//\n')
    f.write('// Multi-head self-attention: Y[%dx%d] = concat_h(softmax(Q_h @ Kt_h / sqrt(%d)) @ V_h) @ Wo,\n'
            % (s_size, e_size, p_size))
    f.write('// Q = X @ Wq, Kt = Wk^T @ X^T, V = X @ Wv, %d heads of size %d\n' % (n_heads, p_size))
    f.write('\n')
    f.write('#ifndef _TEST_MHSA_INCLUDE_GUARD_\n')
    f.write('#define _TEST_MHSA_INCLUDE_GUARD_\n')
    f.write('\n')
    f.write('#define SEQ_LENGTH (%d)\n' % s_size)
    f.write('#define EMBED_SIZE (%d)\n' % e_size)
    f.write('#define PROJ_SIZE  (%d)\n' % p_size)
    f.write('#define NUM_HEADS  (%d)\n' % n_heads)
    f.write('\n')
    if l2ext:
        f.write('// %.1f MiB of L2 arrays: in .l2ext, above the default L2 (build and run gvsoc with\n' % (l2_bytes / 2**20))
        f.write('// l2_size >= 0x10000000)\n')
        f.write('#define L2EXT __attribute__((section(".l2ext")))\n')
    else:
        f.write('#define L2EXT\n')
    f.write('\n')
    write_fp16_array(f, 'x_inp  [SEQ_LENGTH * EMBED_SIZE] L2EXT', X)
    write_fp16_array(f, 'xt_inp [EMBED_SIZE * SEQ_LENGTH] L2EXT', Xt)
    write_fp16_array(f, 'wq_inp [EMBED_SIZE * NUM_HEADS * PROJ_SIZE] L2EXT', Wq)
    write_fp16_array(f, 'wkt_inp[NUM_HEADS * PROJ_SIZE * EMBED_SIZE] L2EXT', Wkt)
    write_fp16_array(f, 'wv_inp [EMBED_SIZE * NUM_HEADS * PROJ_SIZE] L2EXT', Wv)
    write_fp16_array(f, 'wo_inp [NUM_HEADS * PROJ_SIZE * EMBED_SIZE] L2EXT', Wo)
    # intermediates, written by the test
    f.write('uint16_t q_l2 [SEQ_LENGTH * NUM_HEADS * PROJ_SIZE] L2EXT __attribute__((aligned(64))) = {0};\n')
    f.write('uint16_t kt_l2[NUM_HEADS * PROJ_SIZE * SEQ_LENGTH] L2EXT __attribute__((aligned(64))) = {0};\n')
    f.write('uint16_t v_l2 [SEQ_LENGTH * NUM_HEADS * PROJ_SIZE] L2EXT __attribute__((aligned(64))) = {0};\n')
    f.write('uint16_t o_l2 [SEQ_LENGTH * NUM_HEADS * PROJ_SIZE] L2EXT __attribute__((aligned(64))) = {0};\n')
    f.write('uint16_t y_out[SEQ_LENGTH * EMBED_SIZE] L2EXT __attribute__((aligned(64))) = {0};\n')
    f.write('\n')
    # one golden per mesh side, selected by MESH_X_TILES
    f.write('#include "addr_map/tile_config.h"\n')
    f.write('\n')
    for i, mesh_x in enumerate(args.tiles):
        f.write('#%s MESH_X_TILES == %d\n' % ('if' if i == 0 else 'elif', mesh_x))
        write_fp16_array(f, 'y_gold[SEQ_LENGTH * EMBED_SIZE] L2EXT', Y[mesh_x])
    f.write('#else\n')
    f.write('#error "no y_gold for this MESH_X_TILES, regenerate with mhsa_pipeopt4/gen_golden.py --tiles"\n')
    f.write('#endif\n')
    f.write('\n')
    f.write('#endif //_TEST_MHSA_INCLUDE_GUARD_\n')
