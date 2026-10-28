#!/usr/bin/env python3
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""
Generate include/test.h for conv_im2col, a K x K convolution (odd K), stride 1, "same" zero padding
(K - 1) / 2, no bias, HWC:

  Y[H*W x COUT] = im2col(X) [H*W x K*K*CIN] @ W [K*K*CIN x COUT]

with the im2col columns in (ky, kx, ci) order, the order of the rows of W. The golden is bit-true (one RedMulE job
per output element over the whole K*K*CIN reduction, FP16 FMA in that order from 0, the padding zeros included). It
does not depend on the mesh; the block of output pixels of the test (CONV_CP, a divisor of W) is chosen per mesh side:
the largest whose buffers fit L1.

Usage:
  python3 tests/magia/mgc/conv_im2col/gen_golden.py [--h 16 --w 16 --cin 32 --cout 64 --ksize 3]
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'mesh', 'mm_generator'))
from addr_map import l1_bytes  # noqa: E402
from bittrue import matrix_multiply_with_bittrue_fma, write_fp16_array  # noqa: E402

L1_BYTES = l1_bytes() - 64  # L1_SIZE + 1, minus mgc's L1_MARGIN: the mgc buffers have all the rest of the L1

parser = argparse.ArgumentParser("K x K convolution (im2col GEMM) test")
parser.add_argument('--h', type=int, default=16)      # CONV_H
parser.add_argument('--w', type=int, default=16)      # CONV_W
parser.add_argument('--cin', type=int, default=32)    # CONV_CIN
parser.add_argument('--cout', type=int, default=64)   # CONV_COUT
parser.add_argument('--ksize', type=int, default=3)   # CONV_K (odd)
parser.add_argument('--tiles', type=int, nargs='+', default=[2, 4, 8, 16])
parser.add_argument('--cp', type=int, default=None)   # force CONV_CP (output pixels per block, a divisor of W)
parser.add_argument('--nb', type=int, default=None)   # force CONV_NB (blocks of output channels, a divisor of COUT)
parser.add_argument('--seed', type=int, default=42)
# place the L2 arrays in the .l2ext section (above the default L2: run with l2_size >= 0x10000000), by default only
# if they do not fit the default data region (with some room for the rest)
parser.add_argument('--l2ext', choices=['auto', 'yes', 'no'], default='auto')
parser.add_argument('--inc_dir', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'include'))
args = parser.parse_args()

h, w, cin, cout, k = args.h, args.w, args.cin, args.cout, args.ksize
assert k % 2 == 1, "the kernel side must be odd ('same' padding)"
assert h >= k, "the image needs at least ksize rows (the input window is clamped inside it)"
pad = (k - 1) // 2
kkc = k * k * cin


def ceil_div(a, b):
    return (a + b - 1) // b


def schedule(mesh):
    """(CONV_NB, CONV_CP) for a mesh side: the blocks of output channels of the outer blocking loop, 1 doubled
    until some CONV_CP fits, and the largest divisor CONV_CP of W, whose buffers fit L1 (weights of the tile's output
    channels of a block, and double-buffered input rows, im2col block and output block); None if nothing fits."""
    th = ceil_div(h, mesh)
    if h - (mesh - 1) * th < 1:
        return None
    cps = [args.cp] if args.cp else [c for c in range(w, 0, -1) if w % c == 0]
    nbs = [args.nb] if args.nb else [1 << e for e in range(cout.bit_length()) if cout % (1 << e) == 0]
    for nb in nbs:
        assert cout % nb == 0, "CONV_NB must divide COUT"
        cb = cout // nb
        tc = ceil_div(cb, mesh)
        if cb - (mesh - 1) * tc < 1:  # a tile without channels
            break
        for cp in cps:
            assert w % cp == 0, "CONV_CP must divide W"
            l1 = 2 * (kkc * tc + 2 * k * w * cin + 2 * cp * kkc + 2 * cp * tc)
            if l1 <= L1_BYTES:
                return nb, cp
    return None


sched = {mesh: schedule(mesh) for mesh in args.tiles}
for mesh, sc in sched.items():
    if sc is None:
        print("tiles=%d: no schedule (a tile without rows or channels, or nothing fits L1)" % mesh)
    else:
        print("tiles=%d: CONV_NB=%d (%d output channels per block), CONV_CP=%d (%d block(s) per output row)"
              % (mesh, sc[0], cout // sc[0], sc[1], w // sc[1]))
if all(sc is None for sc in sched.values()):
    raise SystemExit("no mesh side has a schedule")

# X in [0, 1); W positive, so that Y stays away from 0, where check(tol=...) is too strict
torch.manual_seed(args.seed)
X = torch.rand(h, w, cin).to(torch.float16).numpy()
Wt = (torch.rand(k, k, cin, cout) * (2.0 / kkc)).to(torch.float16).numpy()


def im2col(x):
    """[H*W x K*K*CIN], columns in (ky, kx, ci) order, zeros outside the image."""
    xp = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
    taps = [xp[ky:ky + h, kx:kx + w, :] for ky in range(k) for kx in range(k)]
    return np.stack(taps, axis=2).reshape(h * w, kkc)


print("Computing bit-true convolution..")
Xcol = im2col(X)
Wm = Wt.reshape(kkc, cout)
Y = matrix_multiply_with_bittrue_fma(Xcol, Wm, np.float16(0))

ref = torch.nn.functional.conv2d(torch.from_numpy(X.astype(np.float64)).permute(2, 0, 1)[None],
                                 torch.from_numpy(Wt.astype(np.float64)).permute(3, 2, 0, 1), padding=pad)
ref = ref[0].permute(1, 2, 0).reshape(h * w, cout).numpy()
print("Max abs error w.r.t. float64 conv2d: ", np.abs(Y.astype(np.float64) - ref).max())
print("Max FP16 bit distance w.r.t. float64 conv2d: ",
      hex(np.abs(Y.view('H').astype(np.int32) - ref.astype(np.float16).view('H').astype(np.int32)).max()))

l2_bytes = 2 * (h * w * cin + kkc * cout + 2 * h * w * cout)
l2ext = args.l2ext == 'yes' or (args.l2ext == 'auto' and l2_bytes > 12 * 2**20)
print("\nL2 arrays: %.1f MiB, %s" % (l2_bytes / 2**20, 'in .l2ext (needs l2_size >= 0x10000000)' if l2ext
                                    else 'in the default data region'))

os.makedirs(args.inc_dir, exist_ok=True)
with open(os.path.join(args.inc_dir, 'test.h'), 'w') as f:
    f.write('// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.\n')
    f.write('// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n')
    f.write('// SPDX-License-Identifier: Apache-2.0\n')
    f.write('// Generated by conv_im2col/gen_golden.py (H=%d, W=%d, CIN=%d, COUT=%d, K=%d, tiles=%s%s, seed=%d)\n'
            % (h, w, cin, cout, k, ','.join(map(str, args.tiles)), ', cp=%d' % args.cp if args.cp else '', args.seed))
    f.write('//\n')
    f.write('// %dx%d convolution, "same" padding, stride 1, HWC: Y[%dx%dx%d] = conv(X[%dx%dx%d], W[%dx%dx%dx%d])\n'
            % (k, k, h, w, cout, h, w, cin, k, k, cin, cout))
    f.write('\n')
    f.write('#ifndef _TEST_CONV_IM2COL_INCLUDE_GUARD_\n')
    f.write('#define _TEST_CONV_IM2COL_INCLUDE_GUARD_\n')
    f.write('\n')
    f.write('#define CONV_H    (%d)\n' % h)
    f.write('#define CONV_W    (%d)\n' % w)
    f.write('#define CONV_CIN  (%d)\n' % cin)
    f.write('#define CONV_COUT (%d)\n' % cout)
    f.write('#define CONV_K    (%d)\n' % k)
    f.write('#define CONV_PAD  (%d)\n' % pad)
    f.write('// W * CIN (one image row), K * K * CIN (one im2col row), H * W (output pixels)\n')
    f.write('#define CONV_ROW  (%d)\n' % (w * cin))
    f.write('#define CONV_KKC  (%d)\n' % kkc)
    f.write('#define CONV_HW   (%d)\n' % (h * w))
    f.write('\n')
    # schedule per mesh side, selected by MESH_X_TILES
    f.write('#include "addr_map/tile_config.h"\n')
    f.write('\n')
    first = True
    for mesh, sc in sched.items():
        if sc is None:
            continue
        f.write('#%s MESH_X_TILES == %d\n' % ('if' if first else 'elif', mesh))
        f.write('#define CONV_NB (%d)\n' % sc[0])
        f.write('#define CONV_CB (%d)\n' % (cout // sc[0]))
        f.write('#define CONV_CP (%d)\n' % sc[1])
        first = False
    f.write('#else\n')
    f.write('#error "no schedule for this MESH_X_TILES, regenerate with conv_im2col/gen_golden.py --tiles"\n')
    f.write('#endif\n')
    f.write('\n')
    if l2ext:
        f.write('// %.1f MiB of L2 arrays: in .l2ext, above the default L2 (build and run with\n' % (l2_bytes / 2**20))
        f.write('// l2_size >= 0x10000000)\n')
        f.write('#define L2EXT __attribute__((section(".l2ext")))\n')
    else:
        f.write('#define L2EXT\n')
    f.write('\n')
    write_fp16_array(f, 'x_inp [CONV_H * CONV_ROW] L2EXT', X.reshape(h, w * cin))
    write_fp16_array(f, 'w_inp [CONV_KKC * CONV_COUT] L2EXT', Wm)
    write_fp16_array(f, 'y_gold[CONV_HW * CONV_COUT] L2EXT', Y)
    # written by the test
    f.write('uint16_t y_out[CONV_HW * CONV_COUT] L2EXT __attribute__((aligned(64))) = {0};\n')
    f.write('\n')
    f.write('#endif //_TEST_CONV_IM2COL_INCLUDE_GUARD_\n')
