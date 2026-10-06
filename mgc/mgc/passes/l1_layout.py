# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""l1-layout: one L1 layout, identical on every tile.

Buffers and their slots are bump-allocated from `l1_tile_base` in declaration
order. When a buffer is accessed on a neighbour tile (`buf.on(dy, dx)`), the
neighbour's copy must be at the same offset, so slot sizes use the *maximum*
tile sizes (`tile_h_max`): with actual sizes a clipped edge tile would lay its
buffers out differently and receive data at the wrong address.

`buf.on(dy, dx)` resolves to the neighbour's L1 base plus the local offset;
the base is computed once per direction (`l1_base_s` = tile below, ...).
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.passes import ModulePass

from .. import ir
from ..errors import MgcError
from ..expr import Const, Raw, Sym, product
from .common import esize, max_sizes, tensors

DIRS = {(1, 0): 's', (-1, 0): 'n', (0, 1): 'e', (0, -1): 'w'}


@dataclass(frozen=True)
class L1Layout(ModulePass):
    """Assigns an L1 address to every buffer slot and resolves neighbour bases.

    Example: buffers `w` (1 slot, 4 KiB) and `x` (2 slots, 2 KiB each) get
    `l1_tile_base`, `obi_addr_w + 4096`, `obi_addr_x_0 + 2048`. Sets `addrs` on
    each `mg.alloc` and `base` (e.g. `l1_base_s`, the tile below) on each `mg.remote`.
    """
    name = 'l1-layout'

    def apply(self, ctx, module):
        ts = tensors(module)
        remotes = [o for o in module.walk() if isinstance(o, ir.RemoteOp)]
        uniform = bool(remotes)
        prev = None
        for a in [o for o in module.walk() if isinstance(o, ir.AllocOp)]:
            shp = a.view.data.shape
            if uniform:
                shp = [max_sizes(s) for s in shp]
            nbytes = product(*shp, Const(esize(a, ts)))
            depth = a.depth.data
            addrs = []
            for k in range(depth):
                slot = f'obi_addr_{a.sym.data}' if depth == 1 else f'obi_addr_{a.sym.data}_{k}'
                addrs.append(Raw('(l1_tile_base)') if prev is None else
                             Raw(f'{prev[0]} + ({prev[1].c()})'))
                prev = (slot, nbytes)
            a.attributes['addrs'] = ir.exprs(addrs)
        for r in remotes:
            dy, dx = r.dy.data.value(), r.dx.data.value()
            if dy is None or dx is None:
                raise MgcError(r, 'neighbour offsets must be compile-time constants')
            suffix = DIRS.get((dy, dx), f'{dy}_{dx}'.replace('-', 'm'))
            r.attributes['base'] = StringAttr(f'l1_base_{suffix}')
