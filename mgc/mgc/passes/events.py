# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""event-alloc: event storage names + use check.

Storage: one `mg_event_t` per (buffer, direction) for DMA, one per
accelerator for un-pipelined jobs, event arrays as declared. The two
directions of a buffer share `idma_evt_<buf>` unless they can be in flight at
the same time (then `idma_evt_<buf>_in` / `_out`).

Check (dataflow over the region tree, states none/pending/maybe):
* issuing on storage that is certainly still pending is an error (the event
  of the previous transfer/job would be overwritten before being waited);
* waiting on storage that was certainly never issued is an error.
`maybe` (only on some paths, e.g. under correlated `if`s) is accepted, so
the check never rejects a correct program; loops are analysed twice, with
event-array slots rotated across the back edge.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.passes import ModulePass

from .. import ir
from ..errors import MgcError
from ..expr import Sym, const_of, sub

N, P, M = 'none', 'pending', 'maybe'


def _merge(a, b):
    if a is None:
        return b
    if b is None:
        return a
    out = {}
    for k in set(a) | set(b):
        x, y = a.get(k, N), b.get(k, N)
        out[k] = x if x == y else M
    return out


class _Analysis:
    """Dataflow walk over the region tree tracking, per event storage, whether
    it is `none`, `pending` (issued, not waited) or `maybe`.

    Raises on re-issue of a pending event and on waiting a never-issued one;
    also records in `overlap` the buffers whose in/out DMA events can be pending
    simultaneously (they then need separate storage).
    """

    def __init__(self, evarrays):
        """`evarrays`: event-array name -> depth (needed to wrap constant indices)."""
        self.evarrays = evarrays
        self.overlap = set()
        self.loops = []  # loop variables, innermost last
        self.cont = []  # states at `continue`, per loop

    def key(self, e: ir.EvtRef, op):
        """Canonical identity of the storage `e` names (array slots are
        normalised to constant or loop-variable-relative indices)."""
        if e.kind == 'array':
            idx = e.index
            c = const_of(idx)
            if c is not None and not idx.syms():
                return ('arr', e.name, None, c % self.evarrays[e.name])
            syms = idx.syms()
            var = next(iter(syms)) if len(syms) == 1 else None
            return ('arr', e.name, var, const_of(sub(idx, Sym(var), fold=False)) if var else idx.c())
        return (e.kind, e.name, e.dir)

    def note(self, st):
        """Record buffers whose load and store events are pending at once."""
        bufs = {k[1] for k, v in st.items() if k[0] == 'dma' and v != N}
        for b in bufs:
            if st.get(('dma', b, 0), N) != N and st.get(('dma', b, 1), N) != N:
                self.overlap.add(b)

    def issue(self, op, st, e):
        """Mark `e` pending in `st`; error if it already certainly is."""
        k = self.key(e, op)
        if st.get(k, N) == P:
            what = f'`{e.name}`' + (' (out)' if e.dir == 1 else '') if e.kind == 'dma' else f'`{e.name}`'
            raise MgcError(op, f'a previous transfer/job on {what} was not waited before reusing its event')
        st[k] = P
        self.note(st)

    def run(self, ops, st):
        """Walk `ops` from state `st`; returns the state afterwards (None if the
        block always ends in `continue`). `if` branches are merged, loops are
        walked twice to catch events left pending across iterations."""
        for op in ops:
            if isinstance(op, ir.DmaOp):
                self.issue(op, st, op.event.data)
            elif isinstance(op, ir.HwpeOp) and op.action.data in ('enqueue', 'oneshot'):
                self.issue(op, st, op.event.data)
            elif isinstance(op, ir.JobOp) and op.event is not None:
                self.issue(op, st, op.event.data)
            elif isinstance(op, ir.WaitOp):
                k = self.key(op.event.data, op)
                if st.get(k, N) == N:
                    raise MgcError(op, f'wait on an event that was never issued (`{op.event.data.name}`)')
                st[k] = N
            elif isinstance(op, ir.IfOp):
                a = self.run(op.then.block.ops, dict(st))
                b = self.run(op.else_.block.ops, dict(st))
                st = _merge(a, b)
                if st is None:
                    return None  # both branches left the block
            elif isinstance(op, ir.ForOp):
                var = op.var.data
                lo = const_of(op.lo.data) or 0
                entry = {}
                for k, v in st.items():
                    # constant array slots become relative to the loop variable
                    if k[0] == 'arr' and k[2] is None:
                        entry[('arr', k[1], var, k[3] - lo)] = v
                    else:
                        entry[k] = v
                self.loops.append(var)
                self.cont.append([])
                s1 = self.run(op.body.block.ops, dict(entry))
                for c in self.cont[-1]:
                    s1 = _merge(s1, c)
                self.cont[-1] = []
                s2 = self.run(op.body.block.ops, _merge(entry, self.rotate(s1, var) if s1 else None))
                for c in self.cont[-1]:
                    s2 = _merge(s2, c)
                self.loops.pop()
                self.cont.pop()
                out = _merge(st, self.rotate(s2, var) if s2 else None)
                st = {k: v for k, v in out.items() if not (k[0] == 'arr' and k[2] == var)}
            elif isinstance(op, ir.ContinueOp):
                if self.cont:
                    self.cont[-1].append(dict(st))
                return None  # the rest of the block is unreachable
        return st

    @staticmethod
    def rotate(st, var):
        """Re-express event-array slots after one loop iteration: slot `v + k`
        becomes `v + k - 1` relative to the next value of `v`."""
        out = {}
        for k, v in st.items():
            if k[0] == 'arr' and k[2] == var and isinstance(k[3], int):
                out[('arr', k[1], var, k[3] - 1)] = v
            else:
                out[k] = v
        return out


@dataclass(frozen=True)
class EventAlloc(ModulePass):
    """Checks event usage and assigns each event its C storage.

    Example: DMA loads and stores of buffer `y` never overlapping share
    `&idma_evt_y`; if they can be in flight together they get `&idma_evt_y_in`
    and `&idma_evt_y_out`. Redmule jobs use `&redmule_evt`.
    """
    name = 'event-alloc'

    def apply(self, ctx, module):
        evarrays = {o.sym.data: o.depth.data for o in module.walk() if isinstance(o, ir.EventsOp)}
        tiles = next(o for o in module.walk() if isinstance(o, ir.TilesOp))
        an = _Analysis(evarrays)
        an.run(tiles.body.block.ops, {})

        def cname(e: ir.EvtRef):
            if e.kind == 'dma':
                base = f'idma_evt_{e.name}'
                if e.name in an.overlap:
                    base += '_in' if e.dir == 0 else '_out'
                return '&' + base
            if e.kind == 'hwpe':
                name = f'{e.name}_evt'
                return '&' + (name + '_job' if name in evarrays else name)
            return e.cname

        for op in module.walk():
            ev = op.attributes.get('event')
            if isinstance(ev, ir.EvtAttr):
                e = ev.data
                op.attributes['event'] = ir.EvtAttr(ir.EvtRef(e.kind, e.name, e.dir, e.index, cname(e)))
