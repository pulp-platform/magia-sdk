# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""emit-c: low-level mg IR -> mglib C, in the style of the hand-written tests.

The emitter makes no decisions: every address, event, parameter and layout was
resolved by the passes. Each buffer's fixed transfer descriptor
(`len_/std_/reps_/obi_addr_/axi_addr_`) is printed once after its allocation
and referred to by name in the transfers (a naming convention for
readability, not a transformation).
"""

from __future__ import annotations

import inspect
import os

from xdsl.dialects.builtin import ModuleOp

from . import devices
from . import ir
from .errors import MgcError
from .expr import Const, Expr, is_zero
from .passes.common import alloc_of, kernels, tensors
from .passes.multibuffer import ORDER, ptr_name

WAIT_MODE = 'WAIT_MODE'


class Writer:
    COLS = 100

    def __init__(self):
        self.lines = []
        self.ind = 1

    def line(self, s=''):
        self.lines.append(('    ' * self.ind + s) if s else '')

    def blank(self):
        last = self.lines[-1].strip() if self.lines else ''
        # no blank line at block start or between a comment and what it describes
        if last and not last.endswith('{') and not last.startswith('//') and last != '*/' \
                and not last.startswith('case '):
            self.lines.append('')

    def block_comment(self, text):
        self.blank()
        self.line('/**')
        for l in inspect.cleandoc(text).splitlines():
            self.line((' * ' + l).rstrip())
        self.line(' */')

    def call(self, fn, args):
        one = f'{fn}({", ".join(args)});'
        if len('    ' * self.ind + one) <= self.COLS:
            self.line(one)
            return
        pad = ' ' * (len(fn) + 1)
        self.line(f'{fn}({args[0]},')
        for a in args[1:-1]:
            self.line(f'{pad}{a},')
        self.line(f'{pad}{args[-1]});')


def _paren(e: Expr) -> str:
    return e.c() if e.prec() == 3 else f'({e.c()})'


DECLARATIVE = (ir.SplitOp, ir.AllocOp, ir.EventsOp)


class Emitter:

    def __init__(self, module: ModuleOp, filename: str):
        self.m = module
        self.filename = filename
        self.ts = tensors(module)
        self.kdefs = kernels(module)
        self.w = Writer()
        self.test = next(o for o in module.ops if isinstance(o, ir.TestOp))
        self.tiles = next(o for o in self.test.body.block.ops if isinstance(o, ir.TilesOp))
        self.evkinds = {o.sym.data: o.kind.data for o in module.walk() if isinstance(o, ir.EventsOp)}
        self.uses_sync = any(isinstance(o, ir.SyncOp) for o in module.walk())
        checks = [o for o in module.walk() if isinstance(o, ir.CheckOp)]
        if len(checks) > 1:
            raise MgcError(checks[1], 'only one check() per test')
        self.check_top = None
        if checks:
            self.check_top = checks[0]
            while self.check_top.parent_op() is not self.tiles:
                self.check_top = self.check_top.parent_op()

    # -- statements -----------------------------------------------------------
    def block(self, ops):
        ops = list(ops)
        k = 0
        while k < len(ops):
            op = ops[k]
            if isinstance(op, ir.SplitOp):
                j = k
                while j < len(ops) and isinstance(ops[j], ir.SplitOp):
                    j += 1
                self.splits(ops[k:j])
                k = j
                continue
            if op is self.check_top:
                self.w.line('uint32_t errors = 0;')
            self.op(op)
            k += 1

    def op(self, op):
        w = self.w
        if isinstance(op, ir.CommentOp):
            if op.style.data == 'block':
                w.block_comment(op.text.data)
            else:
                w.blank()
                w.line('// ' + op.text.data)
        elif isinstance(op, ir.ScalarOp):
            if op.decl.data:
                w.line(f'{op.ctype.data} {op.sym.data} = {op.value.data.c()};')
            else:
                w.line(f'{op.sym.data} = {op.value.data.c()};')
        elif isinstance(op, ir.AllocOp):
            self.alloc(op)
        elif isinstance(op, (ir.EventsOp, ir.RemoteOp)):
            pass
        elif isinstance(op, ir.ForOp):
            v = op.var.data
            w.blank()
            w.line(f'for ({op.ctype.data} {v} = {op.lo.data.c()}; {v} < {op.hi.data.c()}; {v}++) {{')
            w.ind += 1
            self.block(op.body.block.ops)
            w.ind -= 1
            w.line('}')
        elif isinstance(op, ir.IfOp):
            w.blank()
            self.if_(op)
        elif isinstance(op, ir.ContinueOp):
            w.line('continue;')
        elif isinstance(op, ir.DmaOp):
            self.dma(op)
        elif isinstance(op, ir.HwpeOp):
            self.hwpe(op)
        elif isinstance(op, ir.CallOp):
            self.call(op)
        elif isinstance(op, ir.WaitOp):
            self.wait(op)
        elif isinstance(op, ir.SyncOp):
            w.line(f'{devices.SYNC[op.scope.data]}(&fsync_ctrl);')
            w.line(f'eu_fsync_wait(&eu_ctrl, {WAIT_MODE});')
        elif isinstance(op, ir.SelectOp):
            self.select(op)
        elif isinstance(op, ir.CheckOp):
            self.check(op)
        else:
            raise MgcError(op, f'internal: {op.name} was not lowered')

    def if_(self, op, chained=False):
        w = self.w
        if not chained:
            w.line(f'if ({op.cond.data.c()}) {{')
        w.ind += 1
        self.block(op.then.block.ops)
        w.ind -= 1
        els = list(op.else_.block.ops)
        if len(els) == 1 and isinstance(els[0], ir.IfOp):
            w.line(f'}} else if ({els[0].cond.data.c()}) {{')
            self.if_(els[0], chained=True)
            return
        if els:
            w.line('} else {')
            w.ind += 1
            self.block(els)
            w.ind -= 1
        w.line('}')

    def splits(self, ops):
        w = self.w
        mesh = {'y': 'MESH_Y_TILES', 'x': 'MESH_X_TILES'}
        axes = {'y': self.tiles.y.data, 'x': self.tiles.x.data}
        for sp in ops:
            ext = _paren(sp.extent.data)
            m = mesh[sp.axis.data]
            w.line(f'uint32_t {sp.sym.data}_max = (({ext} + {m} - 1) / {m});')
        for sp in ops:
            w.line(f'int32_t {sp.sym.data};')
        for sp in ops:
            ext = _paren(sp.extent.data)
            start = f'{sp.sym.data}_max * {axes[sp.axis.data]}'
            w.blank()
            w.line(f'if ((({start}) + {sp.sym.data}_max) > {ext}) {{')
            w.line(f'    {sp.sym.data} = {ext} - ({start});')
            w.line('} else {')
            w.line(f'    {sp.sym.data} = {sp.sym.data}_max;')
            w.line('}')
        w.blank()
        w.line('if (' + ' || '.join(f'{sp.sym.data} < 1' for sp in ops) + ') {')
        w.line('    return 0;')
        w.line('}')

    def alloc(self, op):
        w = self.w
        n = op.sym.data
        x: ir.Xfer = op.xfer.data
        w.line(f'uint32_t len_{n} = {x.len.c()};')
        for d, (reps, std) in enumerate(x.outer):
            sfx = '' if d == 0 else str(d + 1)
            w.line(f'uint32_t std{sfx}_{n} = {std.c()};')
            w.line(f'uint32_t reps{sfx}_{n} = (uint32_t){_paren(reps)};')
        depth = op.depth.data
        for k, a in enumerate(op.addrs.data):
            slot = f'obi_addr_{n}' if depth == 1 else f'obi_addr_{n}_{k}'
            w.line(f'uint32_t {slot} = {a.data.c()};')
        w.line(f'uint32_t axi_addr_{n} = {op.axi.data.c()};')

    def dma(self, op):
        a = alloc_of(op.local)
        n = a.sym.data
        x: ir.Xfer = op.xfer.data
        if x.rank not in devices.IDMA.functions:
            raise MgcError(op, f'no mglib function for {x.rank}-D transfers')
        if op.remote is not None:
            base = op.remote.owner.base.data
            axi = f'{base} + ({op.addr.data} - l1_tile_base)'
            geo = [x.len.c()]
        else:
            axi = op.axi.data.c()
            geo = [f'len_{n}']
            for d in range(len(x.outer)):
                sfx = '' if d == 0 else str(d + 1)
                geo += [f'std{sfx}_{n}', f'reps{sfx}_{n}']
        args = ['&idma_ctrl', '&eu_ctrl', WAIT_MODE, str(op.dir.data), axi, op.addr.data] + geo
        args += [op.event.data.cname, 'NULL']
        self.w.blank()
        self.w.call(devices.IDMA.functions[x.rank], args)

    def hwpe(self, op):
        dev = devices.HWPES[op.acc.data]
        act = op.action.data
        fn = dev.fn[act].format(job=op.job.data)
        if act in ('commit', 'start', 'commit_start'):
            self.w.line(f'{fn}({dev.ctrl});')
            return
        spec = dev.jobs[op.job.data]
        params = [f'({t}){_paren(p.data)}' for (t, _), p in zip(spec.params, op.params.data)]
        args = [dev.ctrl, '&eu_ctrl', WAIT_MODE] + [a.data for a in op.addrs.data] + params
        args += [op.event.data.cname, 'NULL']
        self.w.blank()
        self.w.call(fn, args)

    def call(self, op):
        k = self.kdefs[op.kernel.data]
        args = [a.data for a in op.addrs.data] + [p.data.c() for p in op.params.data]
        self.w.blank()
        self.w.call(k.cname.data, args)

    def wait(self, op):
        e = op.event.data
        if e.kind == 'dma':
            self.w.line(f'mg_idma_wait(&eu_ctrl, {e.dir}, {WAIT_MODE}, {e.cname});')
        else:
            acc = e.name if e.kind == 'hwpe' else self.evkinds[e.name]
            self.w.line(f'{devices.HWPES[acc].fn["wait"]}(&eu_ctrl, {WAIT_MODE}, {e.cname});')

    def select(self, op):
        w = self.w
        var, depth = op.var.data, op.depth.data

        def lines(c):
            for e in op.entries.data:
                for k in e.offsets:
                    s = (c + k) % depth
                    if e.kind == 'buf':
                        w.line(f'{ptr_name("buf", e.name, k)} = obi_addr_{e.name}_{s};')
                    else:
                        w.line(f'{ptr_name("evt", e.name, k)} = &{e.name}_{s};')

        w.blank()
        if depth == 2:
            w.line(f'if ({var} % 2) {{')
            w.ind += 1
            lines(1)
            w.ind -= 1
            w.line('} else {')
            w.ind += 1
            lines(0)
            w.ind -= 1
            w.line('}')
            return
        w.line(f'switch ({var} % {depth}) {{')
        for c in range(depth):
            w.line(f'case {c}:')
            w.ind += 1
            lines(c)
            w.line('break;')
            w.ind -= 1
        w.line('}')

    def check(self, op):
        w = self.w
        a, b = op.a.data, op.b.data
        ta, tb = self.ts[a.tensor], self.ts[b.tensor]
        ct = ta.ctype
        idx = lambda t: f'(i * {t.stride(0).c()} + j)'
        ra = f'*(volatile {ct} *)({ta.cname} + {idx(ta)})'
        rb = f'*(volatile {ct} *)({tb.cname} + {idx(tb)})'

        def bounds(v, d):
            lo, sz = a.starts[d], a.sizes[d]
            if is_zero(lo):
                return f'{v} = 0; {v} < {sz.c()}'
            return f'{v} = ({lo.c()}); {v} < ({lo.c()} + {_paren(sz)})'

        w.line(f'{ct} computed, expected, diff = 0;')
        w.line(f'for (int {bounds("i", 0)}; i++) {{')
        w.ind += 1
        w.line(f'for (int {bounds("j", 1)}; j++) {{')
        w.ind += 1
        w.line(f'computed = {ra};')
        w.line(f'expected = {rb};')
        w.line('diff = (computed > expected) ? (computed - expected) : (expected - computed);')
        w.line(f'if (diff > 0x{op.tol.data:04x}) {{')
        w.lines.append('#if EVAL == 1')
        w.ind += 1
        w.call('printf', [f'"Error detected at coordinates[%d][%d]: {a.tensor}=%x {b.tensor}=%x\\n"',
                          'i', 'j', ra, rb])
        w.lines.append('#endif')
        w.line('errors++;')
        w.ind -= 1
        w.line('}')
        w.ind -= 1
        w.line('}')
        w.ind -= 1
        w.line('}')
        w.line('printf("TILE %d: Number of errors: %d\\n", hartid, errors);')

    # -- declarations ------------------------------------------------------
    def declarations(self):
        d = ['']
        ind = '    '
        sel = [o for o in self.m.walk() if isinstance(o, ir.SelectOp)]
        bufptr, evptr = [], {}
        for s in sel:
            for e in s.entries.data:
                for k in e.offsets:
                    if e.kind == 'buf':
                        if (e.name, k) not in bufptr:
                            bufptr.append((e.name, k))
                    else:
                        evptr.setdefault(e.name, [])
                        if k not in evptr[e.name]:
                            evptr[e.name].append(k)
        order = [a.sym.data for a in self.m.walk() if isinstance(a, ir.AllocOp)]
        bufptr.sort(key=lambda nk: (ORDER.index(nk[1]), order.index(nk[0])))
        for n, k in bufptr:
            d.append(f'{ind}volatile uint32_t {ptr_name("buf", n, k)};')
        if bufptr:
            d.append('')
        names = {}
        for o in self.m.walk():
            ev = o.attributes.get('event')
            if isinstance(ev, ir.EvtAttr) and ev.data.kind in ('dma', 'hwpe'):
                names.setdefault(ev.data.cname[1:], ev.data)
        names = list(names.items())
        dma = sorted({nm for nm, e in names if e.kind == 'dma'},
                     key=lambda nm: (order.index(next(e.name for x, e in names if x == nm)), nm))
        if dma:
            d.append(f'{ind}mg_event_t ' + ', '.join(dma) + ';')
        for nm in dict.fromkeys(nm for nm, e in names if e.kind == 'hwpe'):
            d.append(f'{ind}mg_event_t {nm};')
        for o in self.m.walk():
            if isinstance(o, ir.EventsOp):
                n, depth = o.sym.data, o.depth.data
                d.append(f'{ind}mg_event_t ' + ', '.join(f'{n}_{k}' for k in range(depth)) + ';')
                if n in evptr:
                    ks = sorted(evptr[n], key=ORDER.index, reverse=True)
                    d.append(f'{ind}mg_event_t ' + ', '.join(f'*{ptr_name("evt", n, k)}' for k in ks) + ';')
        bases = {}
        for o in self.m.walk():
            if isinstance(o, ir.RemoteOp):
                bases[o.base.data] = (o.dy.data.value(), o.dx.data.value())
        for nm, (dy, dx) in bases.items():
            off = ''
            if dy:
                off += f' {"+" if dy > 0 else "-"} ' + ('MESH_X_TILES' if abs(dy) == 1 else f'{abs(dy)} * MESH_X_TILES')
            if dx:
                off += f' {"+" if dx > 0 else "-"} {abs(dx)}'
            d.append(f'{ind}uint32_t {nm} = get_l1_base(hartid{off});')
        return d

    # -- program -----------------------------------------------------------
    def emit(self) -> str:
        ops = list(self.tiles.body.block.ops)
        cut = 0
        for k, op in enumerate(ops):
            if isinstance(op, DECLARATIVE) or (isinstance(op, ir.ScalarOp) and op.decl.data):
                cut = k + 1
        self.block(ops[:cut])
        decl_at = len(self.w.lines)
        self.block(ops[cut:])
        self.w.blank()
        self.w.line('return errors;' if self.check_top is not None else 'return 0;')
        self.w.lines[decl_at:decl_at] = self.declarations()
        return self.render()

    def render(self):
        authors = [a.data for a in self.test.authors.data]
        out = ['// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.',
               '// Licensed under the Apache License, Version 2.0, see LICENSE for details.',
               '// SPDX-License-Identifier: Apache-2.0',
               '//']
        out += [f'// {a}' for a in authors]
        out += ['//',
                f'// Generated by mgc from {os.path.basename(self.filename)}: do not edit, re-run mgc/mgcc.py.',
                '',
                '#include <stdint.h>',
                '#include "test.h"',
                '',
                '#include "tile.h"']
        if self.uses_sync:
            out.append('#include "fsync.h"')
        out += ['#include "mg_idma.h"', '#include "mg_redmule.h"', '#include "mg_event.h"', '']
        defs = [(o.sym.data, o.text.data) for o in self.m.ops if isinstance(o, ir.DefineOp)]
        if WAIT_MODE not in [n for n, _ in defs]:
            defs.append((WAIT_MODE, 'WFE'))
        out += [f'#define {n} {v}' for n, v in defs]
        out.append('')
        for k in self.kdefs.values():
            ops = [o.data.partition(':')[0] for o in k.operands_.data]
            ps = [f'p{i}' for i in range(len(k.params.data))]
            sig = ', '.join(f'uint32_t {x}' for x in ops + ps)
            out.append(f'/** L1 kernel ({k.executor.data}): all pointers are L1 addresses. */')
            out.append(f'extern void {k.cname.data}({sig});')
            out.append('')
        if self.test.doc is not None:
            out.append('/**')
            out += [(' * ' + l).rstrip() for l in inspect.cleandoc(self.test.doc.data).splitlines()]
            out.append(' */')
        out += ['int main(void)', '{']
        pro = PROLOGUE_SYNC if self.uses_sync else PROLOGUE
        out += [('    ' + l) if l else '' for l in pro.format(y=self.tiles.y.data, x=self.tiles.x.data).splitlines()]
        body = self.w.lines
        while body and body[0] == '':
            body.pop(0)
        out += [''] + body + ['}', '']
        res = []
        for l in out:
            if l == '' and res and res[-1] == '':
                continue
            res.append(l)
        return '\n'.join(res)


PROLOGUE = '''/**
 * 0. Get the mesh-tile's hartid, mesh-tile coordinates and define its L1 base,
 * also initialize the controllers for the idma and redmule.
 */
uint32_t hartid = get_hartid();

idma_config_t idma_cfg = {{.hartid = hartid}};
idma_controller_t idma_ctrl = {{
    .base = NULL,
    .cfg  = &idma_cfg,
    .api  = &idma_api,
}};

redmule_config_t redmule_cfg = {{.hartid = hartid}};
redmule_controller_t redmule_ctrl = {{
    .base = NULL,
    .cfg  = &redmule_cfg,
    .api  = &redmule_api,
}};

idma_init(&idma_ctrl);
redmule_init(&redmule_ctrl);

eu_config_t eu_cfg = {{.hartid = hartid}};
eu_controller_t eu_ctrl = {{
    .base = NULL,
    .cfg  = &eu_cfg,
    .api  = &eu_api,
}};
eu_init(&eu_ctrl);
eu_redmule_init(&eu_ctrl, 0);
eu_idma_init(&eu_ctrl, 0);

uint32_t {y} = GET_Y_ID(hartid);
uint32_t {x} = GET_X_ID(hartid);
uint32_t l1_tile_base = get_l1_base(hartid);
'''

PROLOGUE_SYNC = '''/**
 * 0. Get the mesh-tile's hartid, mesh-tile coordinates and define its L1 base,
 * also initialize the controllers for the idma, redmule and fsync
 */
uint32_t hartid = get_hartid();

idma_config_t idma_cfg = {{.hartid = hartid}};
idma_controller_t idma_ctrl = {{
    .base = NULL,
    .cfg  = &idma_cfg,
    .api  = &idma_api,
}};

redmule_config_t redmule_cfg = {{.hartid = hartid}};
redmule_controller_t redmule_ctrl = {{
    .base = NULL,
    .cfg  = &redmule_cfg,
    .api  = &redmule_api,
}};

fsync_config_t fsync_cfg = {{.hartid = hartid}};
fsync_controller_t fsync_ctrl = {{
    .base = NULL,
    .cfg  = &fsync_cfg,
    .api  = &fsync_api,
}};

fsync_init(&fsync_ctrl);
idma_init(&idma_ctrl);
redmule_init(&redmule_ctrl);

eu_config_t eu_cfg = {{.hartid = hartid}};
eu_controller_t eu_ctrl = {{
    .base = NULL,
    .cfg  = &eu_cfg,
    .api  = &eu_api,
}};
eu_init(&eu_ctrl);
eu_redmule_init(&eu_ctrl, 0);
eu_idma_init(&eu_ctrl, 0);
eu_fsync_init(&eu_ctrl, 0);

uint32_t {y} = GET_Y_ID(hartid);
uint32_t {x} = GET_X_ID(hartid);
uint32_t l1_tile_base = get_l1_base(hartid);
'''


def emit(module, filename) -> str:
    return Emitter(module, filename).emit()
