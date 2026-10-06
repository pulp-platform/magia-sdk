# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""pipeline: `mg.pipeline` -> explicit, predicated software pipeline.

Body: DMA transfers and jobs (optionally inside tile-invariant `if`s), written
for one iteration `i`. The pass

1. **assigns stages by dependency hops**: an item's stage is 1 + the stage of
   the latest earlier item that writes a slot it reads (same buffer, same
   index), else 0. For OS: loads 0, gemm 1. For WS: loads 0, gemm 1, store 2.
2. **checks multi-buffer depths**: a slot is live from its first to its last
   stage, so depth >= last - first + 1.
3. **generates** a prologue (stage 0 of iteration 0) and a step loop. In step
   `s`, stage k works on iteration s + 1 - k, guarded by 0 <= s + 1 - k < n.
   Within a step: transfers are issued first (by stage), then jobs; waits come
   last, in issue order (as early as possible / as late as possible).

Accelerator jobs whose inputs are all produced by locally tracked transfers
(no operand written by another tile) in a 2-stage pipeline use the HWPE
*early-enqueue* protocol: the next job is programmed while the DMA runs and
committed behind the current one. Any other job (core kernels, jobs reading
data sent by a neighbour, deeper pipelines) runs inside its step.

`skew=`/`steps=`/`sync=` make it a systolic pipeline: a global time loop of
`steps` iterations in which the tile is active when 0 <= t - skew <= last
step, with a barrier at the end of every time step.

Code shape matters to GCC (it changes register allocation and the cycle count),
so the 2-stage shape is the one of the hand-written tests:
`if (next) {...; wait} else {wait}`; deeper pipelines use per-stage guards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from xdsl.dialects.builtin import ArrayAttr, IntAttr, StringAttr
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter

from .. import devices
from .. import ir
from ..errors import MgcError
from ..expr import Cmp, Const, Expr, Logic, Sym, add, const_of, sub
from .common import alloc_of, kernels, tiles

EVENT_ARRAY = '{acc}_evt'

# ----------------------------------------------------------------------------
# small IR constructors
# ----------------------------------------------------------------------------


def comment(text, style='line'):
    return ir.CommentOp.create(attributes={'text': StringAttr(text), 'style': StringAttr(style)})


def wait(ev: ir.EvtRef):
    return ir.WaitOp.create(attributes={'event': ir.EvtAttr(ev)})


def hwpe(acc, job, action, bufs=(), slots=(), event=None):
    attrs = {'acc': StringAttr(acc), 'job': StringAttr(job), 'action': StringAttr(action),
             'slots': ir.exprs(slots)}
    if event is not None:
        attrs['event'] = ir.EvtAttr(event)
    return ir.HwpeOp.create(operands=list(bufs), attributes=attrs)


def if_(cond, then, els=()):
    return ir.IfOp.create(attributes={'cond': ir.E(cond)},
                          regions=[_region(then), _region(els)])


def _region(ops):
    from xdsl.ir import Block, Region
    return Region(Block(list(ops)))


# ----------------------------------------------------------------------------
# substitution of the iteration variable
# ----------------------------------------------------------------------------


def _subst_attr(a, m):
    if isinstance(a, ir.ExprAttr):
        return ir.ExprAttr(a.data.subst(m))
    if isinstance(a, ir.ViewAttr):
        v = a.data
        return ir.ViewAttr(ir.View(v.tensor, [s.subst(m) for s in v.starts], [s.subst(m) for s in v.sizes],
                                   list(v.kept)))
    if isinstance(a, ir.EvtAttr):
        e = a.data
        return ir.EvtAttr(ir.EvtRef(e.kind, e.name, e.dir, e.index.subst(m) if e.index is not None else None))
    if isinstance(a, ArrayAttr):
        return ArrayAttr([_subst_attr(x, m) for x in a.data])
    return a


def instantiate(op, var, value: Expr):
    """Clone `op` for iteration `var = value`."""
    c = op.clone()
    m = {var: value}
    for o in c.walk():
        for k, a in list(o.attributes.items()):
            o.attributes[k] = _subst_attr(a, m)
        o.lineno = getattr(op, 'lineno', None)
    return c


# ----------------------------------------------------------------------------
# body analysis
# ----------------------------------------------------------------------------


@dataclass(eq=False)
class Item:
    op: object  # DmaOp | JobOp | IfOp (predicated)
    reads: set = field(default_factory=set)
    writes: set = field(default_factory=set)
    events: list = field(default_factory=list)  # (EvtRef, unconditional)
    is_job: bool = False
    stage: int = 0


def _offset(idx: Expr, var: str):
    """Slot index -> ('rel', k) for var + k, ('abs', text) otherwise."""
    if var in idx.syms():
        k = const_of(sub(idx, Sym(var), fold=False))
        if k is None:
            raise MgcError(None, f'slot index `{idx.c()}` must be `{var} + constant`')
        return ('rel', k)
    return ('abs', idx.c())


def _accesses(op, var, kdefs):
    """(reads, writes) of buffer slots for a DMA or job."""
    if isinstance(op, ir.DmaOp):
        key = (alloc_of(op.local).sym.data, _offset(op.slot.data, var))
        return ({key}, set()) if op.dir.data == 1 else (set(), {key})
    roles = _roles(op, kdefs)
    reads, writes = set(), set()
    for v, s, role in zip(op.bufs, op.slots.data, roles):
        key = (alloc_of(v).sym.data, _offset(s.data, var))
        if role in ('in', 'inout'):
            reads.add(key)
        if role in ('out', 'inout'):
            writes.add(key)
    return reads, writes


def _roles(job, kdefs):
    k = job.kernel.data
    if '.' in k:
        acc, j = k.split('.')
        return [r for _, r, _ in devices.HWPES[acc].jobs[j].operands]
    return [o.data.partition(':')[2] for o in kdefs[k].operands_.data]


def _item(op, var, kdefs) -> Item:
    if isinstance(op, (ir.DmaOp, ir.JobOp)):
        r, w = _accesses(op, var, kdefs)
        ev = [] if op.event is None else [(op.event.data, True)]
        return Item(op, r, w, ev, isinstance(op, ir.JobOp))
    if isinstance(op, ir.IfOp):
        if var in op.cond.data.syms():
            raise MgcError(op, f'conditions inside pipeline() must not depend on `{var}` (they predicate whole items)')
        it = Item(op)
        branches = []
        for reg in (op.then, op.else_):
            keys = set()
            for o in reg.block.ops:
                if isinstance(o, (ir.CommentOp, ir.RemoteOp)):
                    continue
                if not isinstance(o, (ir.DmaOp, ir.JobOp)):
                    raise MgcError(o, 'a predicated pipeline item contains only transfers and jobs')
                r, w = _accesses(o, var, kdefs)
                it.reads |= r
                it.writes |= w
                it.is_job |= isinstance(o, ir.JobOp)
                if o.event is not None:
                    keys.add(o.event.data.key())
                    if not any(e.key() == o.event.data.key() for e, _ in it.events):
                        it.events.append((o.event.data, False))
            branches.append(keys)
        both = branches[0] & branches[1]
        it.events = [(e, e.key() in both) for e, _ in it.events]
        return it
    raise MgcError(op, 'pipeline() bodies contain transfers, jobs and tile-invariant ifs')


# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Pipeline(ModulePass):
    name = 'pipeline'

    def apply(self, ctx, module):
        pipes = [op for op in module.walk() if isinstance(op, ir.PipelineOp)]
        if len(pipes) > 1:
            raise MgcError(pipes[1], 'only one pipeline() per test is supported for now')
        for p in pipes:
            self.expand(module, p)

    def expand(self, module, p):
        var = p.var.data
        n = p.n.data
        kdefs = kernels(module)
        docs, items = [], []
        for op in list(p.body.block.ops):
            if isinstance(op, ir.CommentOp):
                if not items:
                    op.detach()
                    docs.append(op)
                continue
            items.append(_item(op, var, kdefs))
        if not items:
            raise MgcError(p, 'empty pipeline() body')

        # 1. stages by dependency hops
        for k, it in enumerate(items):
            prods = [q for q in items[:k] if q.writes & it.reads]
            it.stage = 1 + max(q.stage for q in prods) if prods else 0
        S = 1 + max(it.stage for it in items)
        if S < 2:
            raise MgcError(p, 'nothing to pipeline: no item consumes what another produces')

        # 2. multi-buffer depths
        allocs = {a.sym.data: a for a in module.walk() if isinstance(a, ir.AllocOp)}
        span = {}
        for it in items:
            for b, (kind, _) in it.reads | it.writes:
                if kind == 'rel':
                    lo, hi = span.get(b, (it.stage, it.stage))
                    span[b] = (min(lo, it.stage), max(hi, it.stage))
        for b, (lo, hi) in span.items():
            need = hi - lo + 1
            if allocs[b].depth.data < need:
                raise MgcError(allocs[b], f'multi-buffer `{b}` is live across {need} pipeline stages: '
                               f'declare it with depth={need} or more')

        # 3. job protocol
        remote_written = {alloc_of(r.res).sym.data for r in module.walk() if isinstance(r, ir.RemoteOp)}
        jobs = [it for it in items if it.is_job]
        early = None
        if S == 2 and p.skew is None and len(jobs) == 1 and isinstance(jobs[0].op, ir.JobOp):
            j = jobs[0].op
            k = j.kernel.data
            if '.' in k and not any(alloc_of(v).sym.data in remote_written for v in j.bufs):
                early = jobs[0]

        skew = p.skew.data if p.skew is not None else None
        step = Sym(var, None, 'int32_t' if skew is not None else self._ctype(n, S))
        out = []
        if early is not None:
            out += self.prologue_early(p, items, early, var, module, docs)
            body = self.step_early(items, early, var, step, n)
        else:
            out += self.prologue_generic(p, items, var)
            out += docs
            body = self.step_generic(items, var, step, n, S)

        if skew is None:
            hi = add(n, Const(S - 2)) if S > 2 else n
            loop_body = body
            if p.sync is not None:
                loop_body = body + [self.sync(p)]
            out.append(ir.ForOp.create(attributes={'var': StringAttr(var), 'lo': ir.E(Const(0)),
                                                   'hi': ir.E(hi), 'ctype': StringAttr(step.ctype)},
                                       regions=[_region(loop_body)]))
        else:
            t = p.time.data if p.time is not None else 't'
            steps = p.steps.data
            tctype = 'uint8_t' if (steps.value() is not None and steps.value() <= 0xFF) or \
                (isinstance(steps, Sym) and steps.ctype == 'uint8_t') else 'uint32_t'
            last = add(n, Const(S - 3)) if S >= 3 else sub(n, Const(1))
            active = Logic('&&', [Cmp('>=', step, Const(0)), Cmp('<=', step, last)])
            loop_body = [ir.ScalarOp.create(attributes={'sym': StringAttr(var), 'ctype': StringAttr('int32_t'),
                                                        'value': ir.E(sub(Sym(t), skew)), 'decl': IntAttr(1)}),
                         if_(active, body)]
            if p.sync is not None:
                loop_body.append(self.sync(p))
            out.append(ir.ForOp.create(attributes={'var': StringAttr(t), 'lo': ir.E(Const(0)),
                                                   'hi': ir.E(steps), 'ctype': StringAttr(tctype)},
                                       regions=[_region(loop_body)]))
        for o in out:
            for x in o.walk():
                if getattr(x, 'lineno', None) is None:
                    x.lineno = getattr(p, 'lineno', None)
        Rewriter.insert_op(out, InsertPoint.before(p))
        p.detach()
        p.erase()

    @staticmethod
    def _ctype(n, S):
        hv = n.value()
        small = (hv is not None and hv + S <= 0xFF) or (isinstance(n, Sym) and n.ctype == 'uint8_t')
        return 'uint8_t' if small else 'uint32_t'

    @staticmethod
    def sync(p):
        return ir.SyncOp.create(attributes={'scope': StringAttr(p.sync.data)})

    @staticmethod
    def is_sync(it: Item) -> bool:
        ops = [it.op] if not isinstance(it.op, ir.IfOp) else \
            [o for r in (it.op.then, it.op.else_) for o in r.block.ops]
        return any(isinstance(o, ir.JobOp) and '.' not in o.kernel.data for o in ops)

    # -- guards ------------------------------------------------------------
    @staticmethod
    def guard(k, S, step, n) -> Optional[Expr]:
        """Stage k works on iteration step + 1 - k: needs 0 <= step + 1 - k < n."""
        conds = []
        if k >= 2:
            conds.append(Cmp('>', step, Const(k - 2)))
        if k < S - 1:
            bound = sub(n, Const(1 - k)) if k < 1 else add(n, Const(k - 1))
            conds.append(Cmp('<', step, bound))
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else Logic('&&', conds)

    # -- outer pending events (issued before the pipeline, not yet waited) ---
    @staticmethod
    def outer_pending(p):
        pending = []
        for op in p.parent_block().ops:
            if op is p:
                break
            if isinstance(op, ir.DmaOp):
                pending.append(op.event.data)
            elif isinstance(op, ir.WaitOp):
                pending = [e for e in pending if e.key() != op.event.data.key()]
        return pending

    @staticmethod
    def waits_for(it: Item, pred_op=None):
        """Waits for an item's events (predicated ones inside the item's condition)."""
        out = []
        for e, uncond in it.events:
            w = wait(ir.EvtRef(e.kind, e.name, e.dir, e.index))
            out.append(w if uncond or not isinstance(it.op, ir.IfOp) else if_(it.op.cond.data, [w]))
        return out

    # -- HWPE early-enqueue protocol (2 stages) -------------------------------
    def prologue_early(self, p, items, early, var, module, docs):
        j = early.op
        acc, job = j.kernel.data.split('.')
        evname = EVENT_ARRAY.format(acc=acc)
        self.declare_events(module, evname, acc)
        loads = [it for it in items if it.stage == 0]
        out = [instantiate(it.op, var, Const(0)) for it in loads]
        # wait the job's operands, in operand order (own loads, then outer transfers)
        own = {e.key(): e for it in loads for e, _ in it.events}
        outer = {e.key(): e for e in self.outer_pending(p)}
        for v in j.bufs:
            name = alloc_of(v).sym.data
            for key, e in list(own.items()) + list(outer.items()):
                if e.kind == 'dma' and e.name == name and e.dir == 0:
                    out.append(wait(ir.EvtRef('dma', name, 0)))
                    own.pop(key, None)
                    outer.pop(key, None)
                    break
        jz = instantiate(j, var, Const(0))
        out += [comment('enqueue the first job and commit it: its inputs are already in L1'),
                hwpe(acc, job, 'enqueue', jz.bufs, [s.data for s in jz.slots.data],
                     ir.EvtRef('array', evname, index=Const(0))),
                hwpe(acc, job, 'commit')]
        jz.erase()
        return out + docs

    def step_early(self, items, early, var, step, n):
        j = early.op
        acc, job = j.kernel.data.split('.')
        evname = EVENT_ARRAY.format(acc=acc)
        nxt = add(step, Const(1))
        loads = [it for it in items if it.stage == 0]
        nl = [instantiate(it.op, var, nxt) for it in loads]
        jn = instantiate(j, var, nxt)
        enq = [comment('program the next job while the DMA prefetches its operands'),
               hwpe(acc, job, 'enqueue', jn.bufs, [s.data for s in jn.slots.data],
                    ir.EvtRef('array', evname, index=nxt))]
        jn.erase()
        cur = ir.EvtRef('array', evname, index=step)
        then = [comment('launch the current job (committed in the previous iteration)'),
                hwpe(acc, job, 'start'),
                comment('prefetch the operands of the next iteration')]
        then += nl[:1] + enq + nl[1:]
        then.append(comment('commit & start the next job once its operands have landed'))
        for it in loads:
            then += self.waits_for(it)
        then += [hwpe(acc, job, 'commit_start'), comment('wait for the current job'), wait(cur)]
        els = [comment('last iteration: nothing to prefetch')]
        if n.value() is None or n.value() <= 1:
            # a single job was committed in the prologue but never started
            els.append(if_(Cmp('==', step, Const(0)), [hwpe(acc, job, 'start')]))
        els.append(wait(ir.EvtRef('array', evname, index=step)))
        return [if_(Cmp('<', step, sub(n, Const(1))), then, els)]

    @staticmethod
    def declare_events(module, name, acc):
        t = tiles(module)
        if any(isinstance(o, ir.EventsOp) and o.sym.data == name for o in t.body.block.ops):
            raise MgcError(None, f'`{name}` is used by pipeline(): rename your events() array')
        last = None
        for o in t.body.block.ops:
            if isinstance(o, ir.AllocOp):
                last = o
        ev = ir.EventsOp.create(attributes={'sym': StringAttr(name), 'depth': IntAttr(2),
                                            'kind': StringAttr(acc)})
        if last is None:
            raise MgcError(None, 'pipeline() needs L1 buffers')
        Rewriter.insert_op(ev, InsertPoint.after(last))

    # -- generic (in-step jobs) --------------------------------------------------
    def prologue_generic(self, p, items, var):
        loads = [it for it in items if it.stage == 0]
        out = [instantiate(it.op, var, Const(0)) for it in loads]
        for it in loads:
            out += self.waits_for(it)
        # operands of the jobs still in flight from before the pipeline
        need = {alloc_of(v).sym.data for it in items if it.is_job and isinstance(it.op, ir.JobOp)
                for v in it.op.bufs}
        for e in self.outer_pending(p):
            if e.kind == 'dma' and e.name in need:
                out.append(wait(ir.EvtRef(e.kind, e.name, e.dir)))
        if p.sync is not None:
            out.append(self.sync(p))
        return out

    def step_generic(self, items, var, step, n, S):
        def inst(it):
            return instantiate(it.op, var, add(step, Const(1 - it.stage)))

        transfers = sorted([it for it in items if not it.is_job], key=lambda it: it.stage)
        # asynchronous jobs before synchronous (core) calls: a call occupies the
        # control core, so everything that can run meanwhile must already be issued
        jobs = sorted([it for it in items if it.is_job], key=lambda it: (self.is_sync(it), it.stage))
        seq = []  # (guard, op)
        for it in transfers + jobs:
            seq.append((self.guard(it.stage, S, step, n), inst(it)))
        for it in transfers + jobs:
            for w in self.waits_for(it):
                seq.append((self.guard(it.stage, S, step, n), w))
        guards = {g.c() for g, _ in seq if g is not None}
        if len(guards) <= 1:
            # one guard: the shape of the hand-written 2-stage tests
            g = next((g for g, _ in seq if g is not None), None)
            if g is None:
                return [o for _, o in seq]
            return [if_(g, [o for _, o in seq], [o for gg, o in seq if gg is None])]
        out, cur, grp = [], None, []
        for g, o in seq + [(Const(0), None)]:
            key = g.c() if g is not None else None
            if grp and key != cur:
                out += grp if cur is None else [if_(grp_guard, grp)]
                grp = []
            if o is None:
                break
            if not grp:
                cur, grp_guard = key, g
            grp.append(o)
        return out
