# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""multi-buffer: slot indices -> addresses and pointer rotation.

* constant index `x[1]`          -> the slot's static address `obi_addr_x_1`
* relative index `x[v + k]`      -> a rotating pointer (`x_pt`, `x_pt_next`,
  `x_pt_prev`, ...) selected once per value of `v` by an `mg.select`
  (`if (v % 2)` / `switch (v % N)`).

The same applies to event arrays (`redmule_evt[i + 1]` -> `redmule_evt_next`).

Where the selection goes: at an explicit `rotate(v)` if present; otherwise just
before the first statement, in the innermost block holding every use, that
contains a use (comments directly above it stay attached to it). Pointers are
emitted for offset 0 plus every offset used, matching the hand-written tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter

from .. import ir
from ..errors import MgcError
from ..expr import Sym, const_of, sub
from .common import alloc_of, ancestors, child_in

ORDER = [0, -1, 1, -2, 2, -3, 3]


def ptr_name(kind, name, k):
    if kind == 'buf':
        base = f'{name}_pt'
        return base if k == 0 else base + ('_next' if k > 0 else '_prev') + (str(abs(k)) if abs(k) > 1 else '')
    return f'{name}_' + ('curr' if k == 0 else ('next' if k > 0 else 'prev') + (str(abs(k)) if abs(k) > 1 else ''))


def static_name(kind, name, depth, c):
    if kind == 'buf':
        return f'obi_addr_{name}' if depth == 1 else f'obi_addr_{name}_{c % depth}'
    return f'&{name}_{c % depth}'


@dataclass(frozen=True)
class MultiBuffer(ModulePass):
    name = 'multi-buffer'

    def apply(self, ctx, module):
        evarrays = {o.sym.data: o for o in module.walk() if isinstance(o, ir.EventsOp)}
        allocs = [o for o in module.walk() if isinstance(o, ir.AllocOp)]
        order = {a.sym.data: k for k, a in enumerate(allocs)}
        uses = {}  # var -> list of (op, kind, name, depth, offset)

        def resolve(op, kind, name, depth, idx):
            k = const_of(idx)
            if depth == 1 or (k is not None and not idx.syms()):
                if depth == 1 and k not in (0, None) and kind == 'buf':
                    raise MgcError(op, f'`{name}` is a single buffer: index 0 only')
                return static_name(kind, name, depth, k or 0)
            syms = idx.syms()
            if len(syms) != 1:
                raise MgcError(op, f'slot index `{idx.c()}` must be a constant or `var + constant`')
            var = next(iter(syms))
            off = const_of(sub(idx, Sym(var), fold=False))
            if off is None or abs(off) >= depth:
                raise MgcError(op, f'index must be `{var} + k` with |k| < {depth} (the depth of `{name}`)')
            uses.setdefault(var, []).append((op, kind, name, depth, off))
            return ptr_name(kind, name, off)

        for op in list(module.walk()):
            if isinstance(op, ir.DmaOp):
                a = alloc_of(op.local)
                op.attributes['addr'] = StringAttr(resolve(op, 'buf', a.sym.data, a.depth.data, op.slot.data))
            elif isinstance(op, (ir.HwpeOp, ir.CallOp, ir.JobOp)) and len(op.bufs):
                names = []
                for v, s in zip(op.bufs, op.slots.data):
                    a = alloc_of(v)
                    names.append(resolve(op, 'buf', a.sym.data, a.depth.data, s.data))
                op.attributes['addrs'] = ArrayAttr([StringAttr(x) for x in names])
            for attr in ('event',):
                ev = op.attributes.get(attr) if hasattr(op, 'attributes') else None
                if isinstance(ev, ir.EvtAttr) and ev.data.kind == 'array':
                    e = ev.data
                    if e.name not in evarrays:
                        raise MgcError(op, f'undefined event array `{e.name}`')
                    d = evarrays[e.name].depth.data
                    cname = resolve(op, 'evt', e.name, d, e.index)
                    op.attributes[attr] = ir.EvtAttr(ir.EvtRef(e.kind, e.name, e.dir, e.index, cname))

        markers = {o.var.data: o for o in module.walk() if isinstance(o, ir.SelectOp)}
        for var, us in uses.items():
            by_depth = {}
            for op, kind, name, depth, off in us:
                ent = by_depth.setdefault(depth, {})
                ent.setdefault((kind, name), set()).add(off)
            selects = []
            for depth in sorted(by_depth):
                ents = by_depth[depth]
                keys = sorted(ents, key=lambda kn: (kn[0] != 'buf', order.get(kn[1], 0), kn[1]))
                entries = [ir.SelectEntry(kind, name, sorted(ents[(kind, name)] | {0}, key=ORDER.index))
                           for kind, name in keys]
                selects.append(ir.SelectOp.create(attributes={
                    'var': StringAttr(var), 'depth': IntAttr(depth), 'entries': ir.SelectAttr(entries)}))
            if var in markers:
                m = markers.pop(var)
                Rewriter.insert_op(selects, InsertPoint.before(m))
                m.detach()
                m.erase()
                continue
            # innermost block holding every use
            paths = [ancestors(op) for op, *_ in us]
            common = None
            for blk in paths[0]:
                if all(blk in p for p in paths[1:]):
                    common = blk
                    break
            first = min((child_in(common, op) for op, *_ in us), key=lambda o: common.get_operation_index(o))
            prev = first.prev_op
            while isinstance(prev, ir.CommentOp):
                first, prev = prev, prev.prev_op
            Rewriter.insert_op(selects, InsertPoint.before(first))
        for var, m in markers.items():
            raise MgcError(m, f'rotate({var}): no multi-buffer is indexed by `{var}`')
