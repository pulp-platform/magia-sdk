# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""Helpers shared by the passes."""

from __future__ import annotations

import ast
from typing import Dict, List

from xdsl.dialects.builtin import ArrayAttr, ModuleOp
from xdsl.ir import Block, Operation, Region

from .. import ir
from ..errors import MgcError
from ..expr import Expr, Sym, product
from ..frontend import DTYPES, Tensor


def tensors(module: ModuleOp) -> Dict[str, Tensor]:
    out = {}
    for op in module.ops:
        if isinstance(op, ir.TensorOp):
            out[op.sym.data] = Tensor(op.sym.data, op.cname.data, [a.data for a in op.shape.data],
                                      op.dtype.data)
    return out


def kernels(module: ModuleOp) -> Dict[str, ir.KernelOp]:
    return {op.sym.data: op for op in module.ops if isinstance(op, ir.KernelOp)}


def tiles(module: ModuleOp) -> ir.TilesOp:
    for op in module.walk():
        if isinstance(op, ir.TilesOp):
            return op
    raise MgcError(None, 'no tiles() region')


def allocs(module: ModuleOp) -> List[ir.AllocOp]:
    return [op for op in module.walk() if isinstance(op, ir.AllocOp)]


def alloc_of(value) -> ir.AllocOp:
    op = value.owner
    if isinstance(op, ir.RemoteOp):
        op = op.buf.owner
    return op


def esize(alloc: ir.AllocOp, ts) -> int:
    return DTYPES[ts[alloc.view.data.tensor].dtype][1]


def shape(alloc: ir.AllocOp) -> List[Expr]:
    return alloc.view.data.shape


def ancestors(op: Operation) -> List[Block]:
    """Blocks enclosing `op`, innermost first."""
    out = []
    blk = op.parent_block()
    while blk is not None:
        out.append(blk)
        parent = blk.parent_op()
        blk = parent.parent_block() if parent is not None else None
    return out


def child_in(block: Block, op: Operation) -> Operation:
    """The op of `block` that contains (or is) `op`."""
    while op.parent_block() is not block:
        op = op.parent_op()
    return op


def new_block_region(ops) -> Region:
    return Region(Block(list(ops)))


# -- tiny evaluator for descriptor / kernel parameter expressions ---------------


class _Shape:

    def __init__(self, shp, es):
        self.shape = shp
        self.size = product(*shp)
        self.bytes = product(*shp, es)


def eval_shape_expr(text: str, operands: Dict[str, '_Shape'], node=None) -> Expr:
    """Evaluate e.g. "x.shape[0]" or "src.size // 2" over operand shapes."""

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            from ..expr import Const
            return Const(n.value)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in operands:
            if n.attr in ('size', 'bytes'):
                return getattr(operands[n.value.id], n.attr)
        if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute) and n.value.attr == 'shape'
                and isinstance(n.value.value, ast.Name) and n.value.value.id in operands
                and isinstance(n.slice, ast.Constant)):
            shp = operands[n.value.value.id].shape
            if n.slice.value >= len(shp):
                raise MgcError(node, f'{text}: operand has rank {len(shp)}')
            return shp[n.slice.value]
        if isinstance(n, ast.BinOp):
            from ..expr import BinOp, sub
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return sub(a, b)
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.FloorDiv):
                return BinOp('/', a, b)
        raise MgcError(node, f'cannot evaluate parameter expression {text!r}')

    return ev(ast.parse(text, mode='eval'))


def operand_shapes(op, names, ts) -> Dict[str, _Shape]:
    out = {}
    for n, v in zip(names, op.bufs):
        a = alloc_of(v)
        out[n] = _Shape(shape(a), esize(a, ts))
    return out


def max_sizes(e: Expr) -> Expr:
    """Replace tile-split sizes (int32 `tile_h`) by their maximum (`tile_h_max`)."""
    return e.subst({s: Sym(s + '_max') for s in _split_syms(e)})


def _split_syms(e: Expr):
    out = set()

    def walk(x):
        if isinstance(x, Sym) and x.ctype == 'int32_t':
            out.add(x.name)
        for f in ('a', 'b'):
            if hasattr(x, f) and isinstance(getattr(x, f), Expr):
                walk(getattr(x, f))

    walk(e)
    return out
