# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""legalize-dma: tensor views -> iDMA transfers.

The only pass that knows the iDMA's capabilities (devices.IDMA). A view is
turned into a contiguous run of `len` bytes repeated over outer dimensions
(innermost first), after folding dimensions that are contiguous with their
inner neighbour. The result must fit `IDMA.max_rank`; the L1 side is dense.

Each mg.alloc gets the geometry of its declaration view (`xfer`) and its
base L2 address (`axi`). Each mg.dma is checked against its buffer's geometry
and gets its L2 address as the buffer's base plus an offset.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.passes import ModulePass

from .. import devices
from .. import ir
from ..errors import MgcError
from ..expr import BinOp, Const, Paren, Raw, Sym, const_of, equal, is_zero, product, sub
from .common import alloc_of, tensors


def geometry(node, view: ir.View, t, caps=None) -> ir.Xfer:
    """Describe a tensor view as an iDMA transfer (`ir.Xfer`).

    `view` is the slice of L2 tensor `t`; `node` is used for error locations;
    `caps` defaults to `devices.IDMA`. Dimensions of size 1 are dropped and a
    dimension is folded into its inner neighbour when the inner one is a full row.

    Examples for fp16 `X` of shape (M, N):
    - `X[0:8, :]` (full rows)   -> 1-D: `8*N*2` contiguous bytes;
    - `X[0:8, 0:16]` (a block)  -> 2-D: `32` bytes repeated 8 times, L2 stride `N*2`;
    - `X[3, 0:16]`              -> 1-D: 32 bytes.
    Raises `MgcError` if the innermost dimension is strided or more dimensions
    than `caps.max_rank` remain.
    """
    caps = caps or devices.IDMA  # looked up at call time: the table is the extension point
    dims = [[s, t.stride(d)] for d, (s, k) in enumerate(zip(view.sizes, view.kept))
            if k and const_of(s) != 1]
    merged = []
    for s, st in dims:
        # an outer dim folds into the inner one if the inner one is a full row
        if merged and equal(merged[-1][1], product(s, st)):
            merged[-1] = [product(merged[-1][0], s), st]
        else:
            merged.append([s, st])
    if not merged:
        merged = [[Const(1), Const(1)]]
    if const_of(merged[-1][1]) != 1:
        raise MgcError(node, 'the innermost transferred dimension must be contiguous')
    if len(merged) > caps.max_rank:
        raise MgcError(node, f'mglib iDMA transfers are at most {caps.max_rank}-D; this slice needs '
                       f'{len(merged)} non-contiguous dimensions')
    es = Const(t.esize)
    inner = merged[-1][0]
    outer = [(r, product(st, es)) for r, st in reversed(merged[:-1])]
    return ir.Xfer(product(inner, es), outer)


def same_xfer(a: ir.Xfer, b: ir.Xfer) -> bool:
    """True if two transfers have identical rank, length, repetitions and strides."""
    return (a.rank == b.rank and equal(a.len, b.len)
            and all(equal(r1, r2) and equal(s1, s2) for (r1, s1), (r2, s2) in zip(a.outer, b.outer)))


def base_addr(view: ir.View, t) -> BinOp:
    """L2 byte address of the first element of `view`, as a C expression:
    `(uint32_t)x_inp + (row_start * N * 2) + (col_start * 2)`."""
    e = Raw(f'(uint32_t){t.cname}')
    for d, s in enumerate(view.starts):
        if not is_zero(s):
            e = BinOp('+', e, Paren(product(s, t.stride(d), Const(t.esize))))
    return e


@dataclass(frozen=True)
class LegalizeDma(ModulePass):
    """Gives every `mg.alloc` and `mg.dma` its transfer geometry and L2 address.

    Example: `x = L1.alloc(X[0:8, 0:16])` gets a 2-D `xfer` and base address
    `l2_addr_x`; a later `dma.load(x, X[8:16, 0:16])` is checked to have the same
    geometry and gets address `l2_addr_x + (8 * N * 2)`. A load with a different
    geometry or shape than the buffer's declaration is an `MgcError`.
    """
    name = 'legalize-dma'

    def apply(self, ctx, module):
        ts = tensors(module)
        for op in module.walk():
            if isinstance(op, ir.AllocOp):
                v = op.view.data
                t = ts[v.tensor]
                op.attributes['xfer'] = ir.XferAttr(geometry(op, v, t))
                op.attributes['axi'] = ir.E(base_addr(v, t))
            elif isinstance(op, ir.DmaOp):
                self.dma(op, ts)

    def dma(self, op, ts):
        """Check and annotate one `mg.dma`: neighbour stores are dense copies of
        the whole buffer; L2 transfers must match the buffer's declared shape and
        geometry, and get the L2 address as base plus per-dimension offset."""
        a = alloc_of(op.local)
        decl = a.view.data
        if op.remote is not None:
            # L1 -> same slot on a neighbour tile: both sides dense
            t = ts[decl.tensor]
            op.attributes['xfer'] = ir.XferAttr(ir.Xfer(product(*decl.shape, Const(t.esize))))
            return
        view = op.l2.data
        t = ts[view.tensor]
        shp, bshp = view.shape, decl.shape
        if len(shp) != len(bshp) or not all(equal(x, y) for x, y in zip(shp, bshp)):
            raise MgcError(op, f'shape mismatch: {view.tensor} slice is {shp}, `{a.sym.data}` is {bshp}')
        g = geometry(op, view, t)
        if not same_xfer(g, a.xfer.data):
            raise MgcError(op, f'transfer geometry differs from the one `{a.sym.data}` was declared with')
        if view.tensor == decl.tensor:
            axi = Sym(f'l2_addr_{a.sym.data}')
            for d, (s0, s1) in enumerate(zip(decl.starts, view.starts)):
                diff = sub(s1, s0, fold=False)
                if is_zero(diff):
                    continue
                if is_zero(s0):
                    diff = s1
                axi = BinOp('+', axi, Paren(product(diff, t.stride(d), Const(t.esize))))
        else:
            axi = base_addr(view, t)
        op.attributes['xfer'] = ir.XferAttr(a.xfer.data)
        op.attributes['axi'] = ir.E(axi)
