# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""The `mg` dialect (xDSL).

Design choices (see ../README.md, "Compiler architecture"):

* Control flow, L1 buffers and operations are IR; integer *index arithmetic*
  is kept as symbolic expressions (`#mg.expr`, see expr.py) over C-level names
  (macros, scalars, loop counters). This is close to MLIR's affine maps with
  named symbols, and lets the emitter reproduce hand-written C expressions.
* L1 buffers are SSA values (`!mg.buf`): every use is reachable from its
  `mg.alloc`, which is what multi-buffer lowering and layout need.
* Completion events are *storage references* (`#mg.evt`), not SSA tokens: the
  explicit form rotates and aliases physical event storage across iterations
  and branches, which SSA cannot express without dummies. Correct use is
  checked by a dataflow analysis instead (passes/events.py).

High-level ops (removed by passes): mg.pipeline, mg.job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr
from xdsl.ir import Data, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (IRDLOperation, attr_def, irdl_attr_definition, irdl_op_definition,
                       opt_attr_def, opt_operand_def, region_def, result_def, traits_def,
                       var_operand_def, operand_def)
from xdsl.traits import NoTerminator

from .expr import Expr

# ----------------------------------------------------------------------------
# Attribute payloads
# ----------------------------------------------------------------------------


@dataclass(eq=False)
class View:
    """A view of an L2 tensor: per-dimension start/size; `kept` is False for an integer index."""
    tensor: str  # mg.tensor symbol
    starts: List[Expr]
    sizes: List[Expr]
    kept: List[bool]

    @property
    def shape(self):
        return [s for s, k in zip(self.sizes, self.kept) if k]

    def __str__(self):
        parts = []
        for st, sz, k in zip(self.starts, self.sizes, self.kept):
            parts.append(f'{st.c()}:+{sz.c()}' if k else st.c())
        return f'{self.tensor}[{", ".join(parts)}]'


@dataclass(eq=False)
class Xfer:
    """A legalized iDMA transfer: contiguous `len` bytes, repeated over `outer`
    dimensions given innermost-first as (reps, L2-side stride in bytes).
    rank = 1 + len(outer); the L1 side is always dense."""
    len: Expr
    outer: List[Tuple[Expr, Expr]] = field(default_factory=list)

    @property
    def rank(self):
        return 1 + len(self.outer)

    def __str__(self):
        o = ''.join(f' x [{r.c()} @ {s.c()}]' for r, s in self.outer)
        return f'{self.len.c()}B{o}'


@dataclass(eq=False)
class EvtRef:
    """Event storage: a DMA event (per buffer and direction), an accelerator's
    default event, or an element of an event array."""
    kind: str  # 'dma' | 'hwpe' | 'array'
    name: str  # buffer name | accelerator | array name
    dir: Optional[int] = None  # dma only
    index: Optional[Expr] = None  # array only
    cname: Optional[str] = None  # resolved C pointer expression (event-alloc / multi-buffer)

    def key(self):
        return (self.kind, self.name, self.dir)

    def __str__(self):
        s = f'{self.kind}:{self.name}'
        if self.dir is not None:
            s += f'/{self.dir}'
        if self.index is not None:
            s += f'[{self.index.c()}]'
        if self.cname:
            s += f' = {self.cname}'
        return s


@dataclass(eq=False)
class SelectEntry:
    """One row of a multi-buffer selection table: for buffer or event `name`, the
    address/storage `offsets` (one per slot) among which `mg.select` picks."""
    kind: str  # 'buf' | 'evt'
    name: str
    offsets: List[int]


# ----------------------------------------------------------------------------
# Attributes and types
# ----------------------------------------------------------------------------


class _PyData(Data[object]):
    """Attribute holding an arbitrary Python object (printed only, never parsed back)."""

    @classmethod
    def parse_parameter(cls, parser):
        raise NotImplementedError('mg attributes are printed for inspection only')

    def print_parameter(self, printer):
        printer.print_string('<"' + str(self.data).replace('"', '\\"') + '">')


@irdl_attr_definition
class ExprAttr(_PyData):
    """`#mg.expr`: a symbolic integer expression (`expr.Expr`), printed as C."""
    name = 'mg.expr'

    def print_parameter(self, printer):
        printer.print_string('<"' + self.data.c() + '">')


@irdl_attr_definition
class ViewAttr(_PyData):
    """`#mg.view`: an L2 tensor `View`, e.g. `W[tile_h_start:+tile_h, 0:+K]`."""
    name = 'mg.view'


@irdl_attr_definition
class XferAttr(_PyData):
    """`#mg.xfer`: a legalized iDMA transfer (`Xfer`), e.g. `128B x [8 @ 256]`."""
    name = 'mg.xfer'


@irdl_attr_definition
class EvtAttr(_PyData):
    """`#mg.evt`: reference to completion-event storage (`EvtRef`)."""
    name = 'mg.evt'


@irdl_attr_definition
class SelectAttr(_PyData):
    """`#mg.select`: the list of `SelectEntry` of a `mg.select` op."""
    name = 'mg.select'

    def print_parameter(self, printer):
        printer.print_string('<"' + '; '.join(f'{e.kind} {e.name} {e.offsets}' for e in self.data) + '">')


@irdl_attr_definition
class BufType(ParametrizedAttribute, TypeAttribute):
    """An L1 buffer (single or multi-buffered), local or on a neighbour tile."""
    name = 'mg.buf'


def E(e: Expr) -> ExprAttr:
    """Wrap an `Expr` as an attribute."""
    return ExprAttr(e)


def S(s: str) -> StringAttr:
    """Wrap a Python string as a `StringAttr`."""
    return StringAttr(s)


def exprs(xs) -> ArrayAttr:
    """List of `Expr` -> array attribute (e.g. a tensor shape or slot indices)."""
    return ArrayAttr([ExprAttr(x) for x in xs])


def strs(xs) -> ArrayAttr:
    """List of strings -> array attribute (e.g. authors, `"name:role"` operands)."""
    return ArrayAttr([StringAttr(x) for x in xs])


# ----------------------------------------------------------------------------
# Operations
# ----------------------------------------------------------------------------


class _Region(IRDLOperation):
    """Base of ops that own regions without needing a terminator."""
    traits = traits_def(NoTerminator())


# -- module level -------------------------------------------------------------


@irdl_op_definition
class TensorOp(IRDLOperation):
    """An L2 tensor living in the test header."""
    name = 'mg.tensor'
    sym = attr_def(StringAttr)
    cname = attr_def(StringAttr)
    shape = attr_def(ArrayAttr)
    dtype = attr_def(StringAttr)


@irdl_op_definition
class DefineOp(IRDLOperation):
    """`#define sym text` owned by the .mgc file (from `define()`)."""
    name = 'mg.define'
    sym = attr_def(StringAttr)
    text = attr_def(StringAttr)


@irdl_op_definition
class KernelOp(IRDLOperation):
    """A software L1 kernel: all pointer operands are L1 buffers."""
    name = 'mg.kernel'
    sym = attr_def(StringAttr)
    cname = attr_def(StringAttr)
    operands_ = attr_def(ArrayAttr)  # "name:role"
    params = attr_def(ArrayAttr)  # expressions over operand shapes
    executor = attr_def(StringAttr)


@irdl_op_definition
class FuncOp(_Region):
    """The generated C function: a standalone test (`kind` = 'test', emitted as `main`)
    or a callable `@top` function (`kind` = 'top'), with its C name, authors,
    docstring and the single `mg.tiles` body."""
    name = 'mg.func'
    kind = attr_def(StringAttr)  # 'test' | 'top'
    fname = attr_def(StringAttr)  # C function name
    test = opt_attr_def(StringAttr)  # test name (kind == 'test')
    authors = attr_def(ArrayAttr)
    doc = opt_attr_def(StringAttr)
    body = region_def()


@irdl_op_definition
class TilesOp(_Region):
    """Level 1 (spatial): the region runs on every mesh tile (SPMD)."""
    name = 'mg.tiles'
    y = attr_def(StringAttr)
    x = attr_def(StringAttr)
    body = region_def()


# -- per tile -----------------------------------------------------------------


@irdl_op_definition
class CommentOp(IRDLOperation):
    """A comment to be reproduced in the C output (`block` = `/** */`, `line` = `//`)."""
    name = 'mg.comment'
    text = attr_def(StringAttr)
    style = attr_def(StringAttr)  # 'block' | 'line'


@irdl_op_definition
class SplitOp(IRDLOperation):
    """Tile split of `extent` over mesh rows or columns (`h = y_id.split(E)`);
    defines the per-tile `sym` (size) and `sym_max` (nominal block size)."""
    name = 'mg.split'
    sym = attr_def(StringAttr)
    axis = attr_def(StringAttr)  # 'y' | 'x'
    extent = attr_def(ExprAttr)


@irdl_op_definition
class ScalarOp(IRDLOperation):
    """Integer variable declaration (`decl`=1) or assignment (`decl`=0): `ctype sym = value;`."""
    name = 'mg.scalar'
    sym = attr_def(StringAttr)
    ctype = attr_def(StringAttr)
    value = attr_def(ExprAttr)
    decl = attr_def(IntAttr)  # 1: declaration, 0: assignment


@irdl_op_definition
class AllocOp(IRDLOperation):
    """L1 buffer with `depth` slots, shaped like `view` (which also fixes its
    transfer geometry). legalize-dma adds `xfer`, l1-layout adds `addrs`."""
    name = 'mg.alloc'
    sym = attr_def(StringAttr)
    view = attr_def(ViewAttr)
    depth = attr_def(IntAttr)
    xfer = opt_attr_def(XferAttr)
    axi = opt_attr_def(ExprAttr)
    addrs = opt_attr_def(ArrayAttr)
    res = result_def(BufType)


@irdl_op_definition
class RemoteOp(IRDLOperation):
    """The same allocation on the tile at (y_id + dy, x_id + dx)."""
    name = 'mg.remote'
    buf = operand_def(BufType)
    dy = attr_def(ExprAttr)
    dx = attr_def(ExprAttr)
    base = opt_attr_def(StringAttr)  # neighbour L1 base variable (l1-layout)
    res = result_def(BufType)


@irdl_op_definition
class EventsOp(IRDLOperation):
    """Array of `depth` accelerator events of the given `kind` (`redmule.events(n)`)."""
    name = 'mg.events'
    sym = attr_def(StringAttr)
    depth = attr_def(IntAttr)
    kind = attr_def(StringAttr)


@irdl_op_definition
class ForOp(_Region):
    """Plain time loop `for (var = lo; var < hi; var++)` over `body`."""
    name = 'mg.for'
    var = attr_def(StringAttr)
    lo = attr_def(ExprAttr)
    hi = attr_def(ExprAttr)
    ctype = attr_def(StringAttr)
    body = region_def()


@irdl_op_definition
class IfOp(_Region):
    """`if (cond) then else else_`; `else_` may be empty."""
    name = 'mg.if'
    cond = attr_def(ExprAttr)
    then = region_def()
    else_ = region_def()


@irdl_op_definition
class ContinueOp(IRDLOperation):
    """`continue;` in the innermost loop."""
    name = 'mg.continue'


@irdl_op_definition
class PipelineOp(_Region):
    """High level: a software-pipelined time loop (see passes/pipeline.py)."""
    name = 'mg.pipeline'
    var = attr_def(StringAttr)
    n = attr_def(ExprAttr)
    skew = opt_attr_def(ExprAttr)
    steps = opt_attr_def(ExprAttr)
    sync = opt_attr_def(StringAttr)
    time = opt_attr_def(StringAttr)
    body = region_def()


@irdl_op_definition
class DmaOp(IRDLOperation):
    """iDMA transfer between a local L1 slot and either an L2 view or the same
    slot on a neighbour tile (`remote`). dir 0: into L1, 1: out of L1."""
    name = 'mg.dma'
    local = operand_def(BufType)
    remote = opt_operand_def(BufType)
    dir = attr_def(IntAttr)
    slot = attr_def(ExprAttr)
    l2 = opt_attr_def(ViewAttr)
    event = attr_def(EvtAttr)
    xfer = opt_attr_def(XferAttr)  # legalize-dma
    axi = opt_attr_def(ExprAttr)  # legalize-dma: L2-side address
    addr = opt_attr_def(StringAttr)  # multi-buffer: local slot address


@irdl_op_definition
class JobOp(IRDLOperation):
    """High level: run `kernel` on L1 operands (an accelerator job such as
    "redmule.gemm" or a software L1 kernel)."""
    name = 'mg.job'
    bufs = var_operand_def(BufType)
    kernel = attr_def(StringAttr)
    slots = attr_def(ArrayAttr)
    event = opt_attr_def(EvtAttr)
    params = opt_attr_def(ArrayAttr)  # job-lower
    addrs = opt_attr_def(ArrayAttr)  # multi-buffer


@irdl_op_definition
class HwpeOp(IRDLOperation):
    """One step of the HWPE protocol: enqueue | oneshot (with operands),
    commit | start | commit_start (without)."""
    name = 'mg.hwpe'
    bufs = var_operand_def(BufType)
    acc = attr_def(StringAttr)
    job = attr_def(StringAttr)
    action = attr_def(StringAttr)
    slots = attr_def(ArrayAttr)
    event = opt_attr_def(EvtAttr)
    params = opt_attr_def(ArrayAttr)
    addrs = opt_attr_def(ArrayAttr)


@irdl_op_definition
class CallOp(IRDLOperation):
    """Synchronous software kernel on the tile's control core."""
    name = 'mg.call'
    bufs = var_operand_def(BufType)
    kernel = attr_def(StringAttr)
    slots = attr_def(ArrayAttr)
    params = opt_attr_def(ArrayAttr)
    addrs = opt_attr_def(ArrayAttr)


@irdl_op_definition
class WaitOp(IRDLOperation):
    """Block until the DMA transfer / accelerator job tied to `event` is done."""
    name = 'mg.wait'
    event = attr_def(EvtAttr)


@irdl_op_definition
class SyncOp(IRDLOperation):
    """Barrier among tiles of `scope` (`GLOBAL`, `ROW` or `COL`)."""
    name = 'mg.sync'
    scope = attr_def(StringAttr)


@irdl_op_definition
class SelectOp(IRDLOperation):
    """Multi-buffer slot selection for index variable `var` (one per depth)."""
    name = 'mg.select'
    var = attr_def(StringAttr)
    depth = opt_attr_def(IntAttr)
    entries = opt_attr_def(SelectAttr)


@irdl_op_definition
class CheckOp(IRDLOperation):
    """Compare L2 view `a` with golden view `b` element-wise, tolerance `tol`."""
    name = 'mg.check'
    a = attr_def(ViewAttr)
    b = attr_def(ViewAttr)
    tol = attr_def(IntAttr)


MG = Dialect('mg', [
    TensorOp, DefineOp, KernelOp, FuncOp, TilesOp, CommentOp, SplitOp, ScalarOp, AllocOp, RemoteOp,
    EventsOp, ForOp, IfOp, ContinueOp, PipelineOp, DmaOp, JobOp, HwpeOp, CallOp, WaitOp, SyncOp,
    SelectOp, CheckOp
], [ExprAttr, ViewAttr, XferAttr, EvtAttr, SelectAttr, BufType])
