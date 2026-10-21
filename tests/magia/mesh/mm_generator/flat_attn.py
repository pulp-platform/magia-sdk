# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <francesco.conti@chips.it>
#
# Golden data for flat_attn_naive:
#   python tests/magia/mesh/mm_generator/flat_attn.py --tiles 2 4 8 16 \
#       --inc_dir tests/magia/mgc/flat_attn_naive/include

# COMPUTE:
# O[s_size, p_size] = softmax( Q[s_size, p_size] @ Kt[p_size, s_size] / sqrt(p_size) ) @ V[s_size, p_size]
#
# The golden is a bit-true FP16 model of the FlatAttention dataflow of flat_attn_naive.mgc
# (online softmax over K/V blocks, each split over the mesh columns), with the RedMulE matmuls
# and the Spatz kernels (Schraudolph exp, sequential reductions) emulated op by op.
# Rows are independent, so the mesh rows and the Q blocking do not change the numerics.
# The mesh columns do (row sum split and binary-tree reduction of O), so there is one golden per
# mesh side, selected by MESH_X_TILES.

import numpy as np
import torch
import argparse
import os
from bittrue import matrix_multiply_with_bittrue_fma, write_fp16_array

parser = argparse.ArgumentParser("FlatAttention Test")
parser.add_argument( '--s_size', type=int, default=256 )  # SEQ_LENGTH
parser.add_argument( '--e_size', type=int, default=64 )   # EMBED_SIZE
parser.add_argument( '--p_size', type=int, default=64 )   # PROJ_SIZE
parser.add_argument( '--n_heads', type=int, default=1 )   # NUM_HEADS
parser.add_argument( '--block_kv', type=int, default=64 ) # S_BLOCK_SIZE_KV in the .mgc
parser.add_argument( '--tiles', type=int, nargs='+', default=[2, 4, 8, 16] )  # mesh sides (make build tiles=N)
parser.add_argument( '--qk_range', type=float, default=2.0 )  # Q, Kt in [-qk_range, qk_range)
parser.add_argument( '--seed', type=int, default=42 )
parser.add_argument( '--inc_dir', type=str, default='include')
args = parser.parse_args()

s_size  = args.s_size
p_size  = args.p_size
block   = args.block_kv
assert s_size % block == 0
for mesh_x in args.tiles:
    assert block % mesh_x == 0
    assert mesh_x & (mesh_x - 1) == 0, "binary-tree reduction of O assumes a power-of-2 mesh"

# Q, Kt in [-qk_range, qk_range): the default 2 gives logits of std ~1.3 after the 1/sqrt(P) scaling,
# i.e. a peaked softmax (with [-1, 1) it is almost uniform and O is ~mean(V)).
# V in [0, 1) keeps O away from 0, where the bitwise check(tol=...) is too strict
torch.manual_seed(args.seed)
r  = args.qk_range
Q  = ((torch.rand(s_size, p_size) * 2 - 1) * r).to(torch.float16).numpy()
Kt = ((torch.rand(p_size, s_size) * 2 - 1) * r).to(torch.float16).numpy()
V  = torch.rand(s_size, p_size).to(torch.float16).numpy()


def f16(x):
    # every Spatz op computes in FP16: exact in float64, then one rounding
    return np.asarray(x, dtype=np.float64).astype(np.float16)

def fast_exp(x):
    # Schraudolph exp of the exp/expminus Spatz kernels: clamp, x*COEF+BIAS, rtz to the FP16 bits
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

def flat_attn(mesh_x):
    tile_kv = block // mesh_x
    o = [np.zeros((s_size, p_size), dtype=np.float16) for _ in range(mesh_x)]
    for sikv in range(s_size // block):
        cols = [slice(sikv * block + jj * tile_kv, sikv * block + (jj + 1) * tile_kv) for jj in range(mesh_x)]

        # attn = Q @ Kt / sqrt(P), per tile column
        attn = [matrix_multiply_with_bittrue_fma(Q, Kt[:, c], np.float16(0)) for c in cols]
        attn = [f16(a.astype(np.float64) * scale) for a in attn]

        # online row max over the mesh row (max is exact, order does not matter)
        m_tile = [a.max(axis=1) for a in attn]
        if sikv > 0:
            m_tile = [np.maximum(m, rowm) for m in m_tile]
        rowm = np.max(m_tile, axis=0)

        attn = [fast_exp(f16(a.astype(np.float64) - rowm[:, None])) for a in attn]
        if sikv > 0:
            alpha = fast_exp(f16(rowmprev.astype(np.float64) - rowm))

        # row sum: per tile, then over the mesh row in jj order on tile (ii, 0)
        rows = rowsum(np.stack([rowsum(a) for a in attn], axis=-1))
        if sikv > 0:
            rows = f16(f16(alpha.astype(np.float64) * rowsprev).astype(np.float64) + rows)
            o = [f16(oj.astype(np.float64) * alpha[:, None]) for oj in o]

        # o += attn @ v, per tile column
        o = [matrix_multiply_with_bittrue_fma(a, V[c, :], oj) for a, c, oj in zip(attn, cols, o)]

        rowmprev, rowsprev = rowm, rows

    # normalize on every tile, then binary-tree reduction over the mesh row
    o = [f16(oj.astype(np.float64) / rows[:, None]) for oj in o]
    stride = 1
    while stride < mesh_x:
        for jj in range(0, mesh_x, 2 * stride):
            o[jj] = f16(o[jj].astype(np.float64) + o[jj + stride])
        stride *= 2
    return o[0]


# Error w.r.t. a float64 reference with the exact exp, for information only
ref = Q.astype(np.float64) @ Kt.astype(np.float64) * scale
ref = np.exp(ref - ref.max(axis=1, keepdims=True))
ref = (ref / ref.sum(axis=1, keepdims=True)) @ V.astype(np.float64)

O = {}
for mesh_x in args.tiles:
    print("\nComputing bit-true FlatAttention, tiles=%d.." % mesh_x)
    O[mesh_x] = flat_attn(mesh_x)
    print("Max abs error w.r.t. float64 reference: ", np.abs(O[mesh_x].astype(np.float64) - ref).max())
    print("Max FP16 bit distance w.r.t. float64 reference: ",
          hex(np.abs(O[mesh_x].view('H').astype(np.int32) - f16(ref).view('H').astype(np.int32)).max()))

print("\n\n")

# ------------------------------------------------------------------------------------#
#                             Header file generation                                  #
# ------------------------------------------------------------------------------------#

# Create include directory if it doesn't exist
inc_path = args.inc_dir
if not os.path.exists(inc_path):
    os.makedirs(inc_path)

f = open(os.path.join(inc_path, 'test.h'), "w")

f.write('// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.\n')
f.write('// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n')
f.write('// SPDX-License-Identifier: Apache-2.0\n')
f.write('// Generated by mm_generator/flat_attn.py (S=%d, P=%d, block_kv=%d, tiles=%s, qk_range=%g, seed=%d)\n'
        % (s_size, p_size, block, ','.join(map(str, args.tiles)), r, args.seed))
f.write('//\n')
f.write('// Flat-Attention: O[%dx%d] = softmax(Q[%dx%d] @ Kt[%dx%d] / sqrt(%d)) @ V[%dx%d]\n'
        % (s_size, p_size, s_size, p_size, p_size, s_size, p_size, s_size, p_size))
f.write('\n')
f.write('#ifndef _TEST_FLAT_ATTN_INCLUDE_GUARD_\n')
f.write('#define _TEST_FLAT_ATTN_INCLUDE_GUARD_\n')
f.write('\n')

f.write('#define SEQ_LENGTH (%d)\n' % s_size)
f.write('#define EMBED_SIZE (%d)\n' % args.e_size)
f.write('#define PROJ_SIZE  (%d)\n' % p_size)
f.write('#define NUM_HEADS  (%d)\n' % args.n_heads)
f.write('\n')

write_fp16_array(f, 'q_inp [SEQ_LENGTH * PROJ_SIZE]', Q)
write_fp16_array(f, 'k_inp [SEQ_LENGTH * PROJ_SIZE]', Kt)
write_fp16_array(f, 'v_inp [SEQ_LENGTH * PROJ_SIZE]', V)
f.write('uint16_t o_out [SEQ_LENGTH * PROJ_SIZE] = {0};\n')
f.write('\n')
# one golden per mesh side, selected by MESH_X_TILES
f.write('#include "addr_map/tile_config.h"\n')
f.write('\n')
for i, mesh_x in enumerate(args.tiles):
    f.write('#%s MESH_X_TILES == %d\n' % ('if' if i == 0 else 'elif', mesh_x))
    write_fp16_array(f, 'o_gold[SEQ_LENGTH * PROJ_SIZE]', O[mesh_x])
f.write('#else\n')
f.write('#error "no o_gold for this MESH_X_TILES, regenerate with mm_generator/flat_attn.py --tiles"\n')
f.write('#endif\n')
f.write('\n')

f.write('#endif //_TEST_FLAT_ATTN_INCLUDE_GUARD_\n')
f.close()
