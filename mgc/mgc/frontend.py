# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""Front-end: .mgc (Python syntax, parsed with `ast`, never executed) -> mg IR.

The front-end only resolves names and builds IR. Legality (DMA geometry,
shapes, event use) is checked by the passes, which report source lines.
"""

from __future__ import annotations

import ast
import inspect
import io
import os
import re
import tokenize
from dataclasses import dataclass
from typing import Dict, List, Optional

from xdsl.dialects.builtin import ArrayAttr, IntAttr, ModuleOp, StringAttr
from xdsl.ir import Block, Region

from . import devices
from . import ir
from .errors import MgcError
from .expr import BinOp, Cmp, Const, Expr, Logic, Not, Sym, const_of, product, sub

DTYPES = {'fp16': ('uint16_t', 2)}
CTYPES = {'u8': ('uint8_t', 0xFF), 'u16': ('uint16_t', 0xFFFF), 'u32': ('uint32_t', 0xFFFFFFFF),
          'i32': ('int32_t', 0x7FFFFFFF)}
ROLES = ('in', 'out', 'inout')

# ----------------------------------------------------------------------------
# Compile-time values bound to Python names
# ----------------------------------------------------------------------------


@dataclass(eq=False)
class Tensor:
    pyname: str
    cname: str
    shape: List[Expr]
    dtype: str

    @property
    def esize(self):
        return DTYPES[self.dtype][1]

    @property
    def ctype(self):
        return DTYPES[self.dtype][0]

    def stride(self, d) -> Expr:
        return product(*self.shape[d + 1:])


@dataclass(eq=False)
class Axis:
    var: Sym
    axis: str  # 'y' | 'x'


@dataclass(eq=False)
class Split:
    name: str
    axis: Axis
    extent: Expr

    @property
    def max(self):
        return Sym(self.name + '_max')

    @property
    def start(self):
        return BinOp('*', self.max, self.axis.var)

    @property
    def size(self):
        return Sym(self.name, None, 'int32_t')


@dataclass(eq=False)
class Buf:
    name: str
    value: object  # SSA value of the mg.alloc
    view: ir.View
    depth: int


@dataclass(eq=False)
class Remote:
    buf: Buf
    dy: Expr
    dx: Expr


@dataclass(eq=False)
class Slot:
    buf: object  # Buf | Remote
    index: Expr


@dataclass(eq=False)
class EvArray:
    name: str
    depth: int
    kind: str


@dataclass(eq=False)
class Kernel:
    sym: str
    cname: str
    operands: List[tuple]  # (name, role)
    params: List[str]
    executor: str


# ----------------------------------------------------------------------------


def parse_header(path):
    defs, text = {}, ''
    if path and os.path.exists(path):
        with open(path) as f:
            text = f.read()
        for m in re.finditer(r'^\s*#define\s+(\w+)\s+\(?\s*(\d+)\s*\)?\s*$', text, re.M):
            defs[m.group(1)] = int(m.group(2))
    return defs, text


def _comments(src):
    out = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT and tok.line.strip().startswith('#'):
            out[tok.start[0]] = tok.string[1:].strip()
    return out


def _name(node):
    return node.id if isinstance(node, ast.Name) else None


def _is_doc(s):
    return isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)


def _call_parts(node):
    """`obj.attr(...)` -> (obj_node, attr); `fn(...)` -> (None, fn)."""
    if not isinstance(node, ast.Call):
        return None, None
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.value, f.attr
    if isinstance(f, ast.Name):
        return None, f.id
    return None, None


class Frontend:

    def __init__(self, src, filename, header=None):
        MgcError.filename = filename
        self.filename = filename
        self.tree = ast.parse(src, filename)
        self.comments = _comments(src)
        self.cursor = 0
        self.hdefs, self.htext = parse_header(header)
        self.header = header
        self.env: Dict[str, object] = {
            'MESH_X_TILES': Sym('MESH_X_TILES'),
            'MESH_Y_TILES': Sym('MESH_Y_TILES'),
        }
        self.module_ops = []
        self.blocks: List[Block] = []

    # -- building ------------------------------------------------------------
    def add(self, op, node=None):
        if node is not None and hasattr(node, 'lineno'):
            op.lineno = node.lineno
        self.blocks[-1].add_op(op)
        return op

    def region(self, stmts, top=False, enter=None):
        blk = Block()
        self.blocks.append(blk)
        if enter:
            enter()
        self.block(stmts, top)
        self.blocks.pop()
        return Region(blk)

    # -- module level --------------------------------------------------------
    def run(self) -> ModuleOp:
        main = None
        for s in self.tree.body:
            if isinstance(s, (ast.Import, ast.ImportFrom)) or _is_doc(s):
                continue
            if isinstance(s, ast.FunctionDef):
                if s.name != 'main':
                    raise MgcError(s, 'an mgc program defines exactly one function, `main`')
                main = s
                continue
            if isinstance(s, ast.Assign):
                self.module_assign(s)
                continue
            raise MgcError(s, 'unsupported statement at module level')
        if main is None:
            raise MgcError(None, 'no `main` function')
        name, authors = 'test', []
        for d in main.decorator_list:
            _, fn = _call_parts(d)
            if fn != 'test':
                raise MgcError(d, 'main must be decorated with @test(...)')
            if d.args:
                name = ast.literal_eval(d.args[0])
            for kw in d.keywords:
                if kw.arg == 'authors':
                    authors = ast.literal_eval(kw.value)
        body = list(main.body)
        doc = body.pop(0).value.value if body and _is_doc(body[0]) else None
        if len(body) != 1 or not isinstance(body[0], ast.For) or _call_parts(body[0].iter)[1] != 'tiles':
            raise MgcError(main, 'main must contain exactly one `for y_id, x_id in tiles():` loop '
                           '(level 1, spatial): all per-tile code goes inside it')
        tl = body[0]
        tgt = tl.target
        if not (isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2):
            raise MgcError(tl, 'tiles() binds two axes: `for y_id, x_id in tiles():`')
        yv, xv = (e.id for e in tgt.elts)
        self.env[yv] = Axis(Sym(yv), 'y')
        self.env[xv] = Axis(Sym(xv), 'x')
        self.cursor = tl.lineno
        self.blocks.append(Block())
        tiles = ir.TilesOp.create(attributes={'y': StringAttr(yv), 'x': StringAttr(xv)},
                                  regions=[self.region(tl.body, top=True)])
        self.blocks.pop()
        attrs = {'test': StringAttr(name), 'authors': ir.strs(authors)}
        if doc:
            attrs['doc'] = StringAttr(doc)
        test = ir.TestOp.create(attributes=attrs, regions=[Region(Block([tiles]))])
        return ModuleOp(self.module_ops + [test])

    def module_assign(self, s):
        _, fn = _call_parts(s.value)
        tgt = s.targets[0]
        if fn == 'sizes':
            names = [ast.literal_eval(a) for a in s.value.args]
            if not isinstance(tgt, ast.Tuple) or len(tgt.elts) != len(names):
                raise MgcError(s, 'unpack sizes(...) into as many names as arguments')
            for t, n in zip(tgt.elts, names):
                if self.header and n not in self.hdefs:
                    raise MgcError(s, f'{n} is not #defined in {self.header}')
                self.env[t.id] = Sym(n, self.hdefs.get(n))
        elif fn == 'l2':
            cname = ast.literal_eval(s.value.args[0])
            shape = [self.expr(e) for e in s.value.args[1].elts]
            dtype = _name(s.value.args[2])
            if dtype not in DTYPES:
                raise MgcError(s, f'unsupported dtype {dtype}; supported: {", ".join(DTYPES)}')
            if self.htext and not re.search(r'\b' + cname + r'\b', self.htext):
                raise MgcError(s, f'L2 tensor {cname} not found in {self.header}')
            t = Tensor(tgt.id, cname, shape, dtype)
            self.env[tgt.id] = t
            self.module_ops.append(ir.TensorOp.create(attributes={
                'sym': StringAttr(tgt.id), 'cname': StringAttr(cname), 'shape': ir.exprs(shape),
                'dtype': StringAttr(dtype)}))
        elif fn == 'define':
            a = s.value.args[0]
            if isinstance(a, ast.Constant):
                val, txt = a.value, str(a.value)
            else:
                val, txt = None, ast.unparse(a)
            self.module_ops.append(ir.DefineOp.create(attributes={'sym': StringAttr(tgt.id),
                                                                   'text': StringAttr(txt)}))
            self.env[tgt.id] = Sym(tgt.id, val if isinstance(val, int) else None)
        elif fn == 'l1_kernel':
            self.l1_kernel(s, tgt.id)
        else:
            raise MgcError(s, 'module level only accepts sizes(), l2(), define() and l1_kernel()')

    def l1_kernel(self, s, sym):
        args = s.value.args
        if len(args) != 1:
            raise MgcError(s, 'l1_kernel("c_function", operands=("name:role", ...), params=(...), executor=CORE)')
        cname = ast.literal_eval(args[0])
        kw = {k.arg: k.value for k in s.value.keywords}
        ops = []
        for o in ast.literal_eval(kw.pop('operands', ast.Tuple([], ast.Load()))):
            n, _, role = o.partition(':')
            if role not in ROLES:
                raise MgcError(s, f'operand {o!r}: role must be one of {", ".join(ROLES)}')
            ops.append((n, role))
        if not ops:
            raise MgcError(s, 'an L1 kernel needs at least one L1 operand')
        params = list(ast.literal_eval(kw.pop('params', ast.Tuple([], ast.Load()))))
        ex = _name(kw.pop('executor', ast.Name('CORE', ast.Load())))
        if ex not in devices.EXECUTORS:
            raise MgcError(s, f'unsupported executor {ex}; supported: {", ".join(devices.EXECUTORS)}')
        if kw:
            raise MgcError(s, f'unknown l1_kernel arguments: {", ".join(kw)}')
        k = Kernel(sym, cname, ops, params, ex)
        self.env[sym] = k
        self.module_ops.append(ir.KernelOp.create(attributes={
            'sym': StringAttr(sym), 'cname': StringAttr(cname),
            'operands_': ir.strs([f'{n}:{r}' for n, r in ops]), 'params': ir.strs(params),
            'executor': StringAttr(ex)}))

    # -- expressions ---------------------------------------------------------
    def ev(self, n):
        if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
            return Const(n.value)
        if isinstance(n, ast.Name):
            if n.id not in self.env:
                raise MgcError(n, f'undefined name `{n.id}`')
            v = self.env[n.id]
            return v.var if isinstance(v, Axis) else v
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return sub(Const(0), self.expr(n.operand))
        if isinstance(n, ast.BinOp):
            a, b = self.ev(n.left), self.ev(n.right)
            if isinstance(a, Split) or isinstance(b, Split):
                raise MgcError(n, 'use .start / .size to do arithmetic on a tile split')
            op = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.FloorDiv: '/', ast.Mod: '%'}.get(type(n.op))
            if op is None or not isinstance(a, Expr) or not isinstance(b, Expr):
                raise MgcError(n, 'unsupported integer expression')
            if op == '+':
                return a + b
            if op == '-':
                return sub(a, b)
            if op == '*':
                return a * b
            return BinOp(op, a, b)
        if isinstance(n, ast.Attribute):
            v = self.ev(n.value)
            if isinstance(v, Split) and n.attr in ('start', 'size'):
                return getattr(v, n.attr)
            raise MgcError(n, f'unknown attribute .{n.attr}')
        if isinstance(n, ast.Call):
            obj, fn = _call_parts(n)
            if fn == 'on' and obj is not None:
                b = self.ev(obj)
                if not isinstance(b, Buf) or len(n.args) != 2:
                    raise MgcError(n, '`buf.on(dy, dx)` names an L1 buffer on a neighbour tile')
                return Remote(b, self.expr(n.args[0]), self.expr(n.args[1]))
            raise MgcError(n, f'`{ast.unparse(n)}` is not an expression')
        if isinstance(n, ast.Subscript):
            v = self.ev(n.value)
            idx = n.slice.elts if isinstance(n.slice, ast.Tuple) else [n.slice]
            if isinstance(v, Tensor):
                return self.view(n, v, idx)
            if isinstance(v, (Buf, Remote, EvArray)):
                if len(idx) != 1:
                    raise MgcError(n, 'multi-buffers and event arrays take a single index')
                i = self.expr(idx[0])
                return ir.EvtRef('array', v.name, index=i) if isinstance(v, EvArray) else Slot(v, i)
            raise MgcError(n, 'only L2 tensors, L1 multi-buffers and event arrays can be indexed')
        if isinstance(n, (ast.Compare, ast.BoolOp)) or (isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not)):
            return self.cond(n)
        raise MgcError(n, f'unsupported expression `{ast.unparse(n)}`')

    def expr(self, n) -> Expr:
        v = self.ev(n)
        if not isinstance(v, Expr):
            raise MgcError(n, f'`{ast.unparse(n)}` is not an integer expression')
        return v

    def cond(self, n) -> Expr:
        if isinstance(n, ast.Compare):
            ops = {ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=', ast.Eq: '==', ast.NotEq: '!='}
            terms, left = [], n.left
            for op, right in zip(n.ops, n.comparators):
                if type(op) not in ops:
                    raise MgcError(n, 'unsupported comparison')
                terms.append(Cmp(ops[type(op)], self.expr(left), self.expr(right)))
                left = right
            return terms[0] if len(terms) == 1 else Logic('&&', terms)
        if isinstance(n, ast.BoolOp):
            return Logic('&&' if isinstance(n.op, ast.And) else '||', [self.cond(v) for v in n.values])
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return Not(self.cond(n.operand))
        return self.expr(n)

    def view(self, n, t: Tensor, idx):
        if len(idx) > len(t.shape):
            raise MgcError(n, f'too many indices for {t.pyname} ({len(t.shape)}-D)')
        starts, sizes, kept = [], [], []
        for d, dim in enumerate(t.shape):
            i = idx[d] if d < len(idx) else ast.Slice(None, None, None)
            if isinstance(i, ast.Slice):
                if i.step is not None and const_of(self.expr(i.step)) != 1:
                    raise MgcError(n, 'iDMA slices must have unit step (mglib has no element stride)')
                lo = self.expr(i.lower) if i.lower else Const(0)
                hi = self.expr(i.upper) if i.upper else dim
                starts.append(lo)
                sizes.append(sub(hi, lo))
                kept.append(True)
            else:
                v = self.ev(i)
                if isinstance(v, Split):
                    starts.append(v.start)
                    sizes.append(v.size)
                    kept.append(True)
                elif isinstance(v, Expr):
                    starts.append(v)
                    sizes.append(Const(1))
                    kept.append(False)
                else:
                    raise MgcError(i, 'index must be an integer, a slice or a tile split')
        return ir.View(t.pyname, starts, sizes, kept)

    # -- statements ----------------------------------------------------------
    def flush_comments(self, s):
        if not hasattr(s, 'lineno'):
            return
        for ln in sorted(self.comments):
            if self.cursor < ln < s.lineno:
                self.add(ir.CommentOp.create(attributes={'text': StringAttr(self.comments[ln]),
                                                         'style': StringAttr('line')}))
        self.cursor = max(self.cursor, s.lineno)

    def block(self, stmts, top=False):
        for s in stmts:
            if self._is_split(s) and not top:
                raise MgcError(s, 'tile splits must be at the top of the tiles() body')
            self.stmt(s)

    @staticmethod
    def _is_split(s):
        return isinstance(s, ast.Assign) and _call_parts(s.value)[1] == 'split'

    def stmt(self, s):
        self.flush_comments(s)
        if _is_doc(s):
            self.add(ir.CommentOp.create(attributes={'text': StringAttr(inspect.cleandoc(s.value.value)),
                                                     'style': StringAttr('block')}), s)
        elif isinstance(s, ast.Expr) and _call_parts(s.value) == (None, 'comment'):
            self.add(ir.CommentOp.create(attributes={'text': StringAttr(s.value.args[0].value),
                                                     'style': StringAttr('line')}), s)
        elif isinstance(s, ast.AnnAssign):
            self.scalar(s, s.target, s.value, _name(s.annotation))
        elif isinstance(s, ast.Assign):
            self.assign(s)
        elif isinstance(s, ast.AugAssign):
            self.scalar(s, s.target, ast.BinOp(ast.Name(_name(s.target), ast.Load()), s.op, s.value), None)
        elif isinstance(s, ast.Expr):
            self.action(s.value, s)
        elif isinstance(s, ast.For):
            self.loop(s)
        elif isinstance(s, ast.If):
            self.add(ir.IfOp.create(attributes={'cond': ir.E(self.cond(s.test))},
                                    regions=[self.region(s.body), self.region(s.orelse)]), s)
        elif isinstance(s, ast.Continue):
            self.add(ir.ContinueOp.create(), s)
        elif isinstance(s, ast.Pass):
            pass
        else:
            raise MgcError(s, f'unsupported statement `{ast.unparse(s).splitlines()[0]}`')
        self.cursor = max(self.cursor, getattr(s, 'end_lineno', 0) or 0)

    def scalar(self, s, tgt, value, ann):
        name = _name(tgt)
        if name is None:
            raise MgcError(s, 'assign to a single name')
        e = self.expr(value)
        old = self.env.get(name)
        if ann is None and isinstance(old, Sym) and old.name == name and not old.ctype.startswith('#'):
            self.add(ir.ScalarOp.create(attributes={'sym': StringAttr(name), 'ctype': StringAttr(old.ctype),
                                                    'value': ir.E(e), 'decl': IntAttr(0)}), s)
            self.env[name] = Sym(name, None, old.ctype)
            return
        if ann is not None:
            if ann not in CTYPES:
                raise MgcError(s, f'unknown type annotation {ann}; use one of {", ".join(CTYPES)}')
            ctype, vmax = CTYPES[ann]
            v = e.value()
            if v is not None and not (0 <= v <= vmax):
                raise MgcError(s, f'{name} = {v} does not fit {ctype}')
        else:
            ctype = 'uint32_t'
        self.add(ir.ScalarOp.create(attributes={'sym': StringAttr(name), 'ctype': StringAttr(ctype),
                                                'value': ir.E(e), 'decl': IntAttr(1)}), s)
        self.env[name] = Sym(name, e.value(), ctype)

    def assign(self, s):
        if len(s.targets) != 1:
            raise MgcError(s, 'chained assignment is not supported')
        tgt = s.targets[0]
        name = _name(tgt)
        obj, fn = _call_parts(s.value)
        on = _name(obj)
        if fn == 'split' and obj is not None:
            ax = self.env.get(on)
            if not isinstance(ax, Axis):
                raise MgcError(s, '.split() applies to a tiles() axis')
            sp = Split(name, ax, self.expr(s.value.args[0]))
            self.env[name] = sp
            self.add(ir.SplitOp.create(attributes={'sym': StringAttr(name), 'axis': StringAttr(ax.axis),
                                                   'extent': ir.E(sp.extent)}), s)
        elif (on, fn) in (('L1', 'alloc'), ('L1', 'multi_buffer')):
            self.alloc(s, name, fn)
        elif fn == 'events' and on in devices.HWPES:
            n = const_of(self.expr(s.value.args[0])) if s.value.args else None
            if not n or n < 1:
                raise MgcError(s, f'{on}.events(n) takes a constant depth >= 1')
            ea = EvArray(name, n, on)
            self.env[name] = ea
            self.add(ir.EventsOp.create(attributes={'sym': StringAttr(name), 'depth': IntAttr(n),
                                                    'kind': StringAttr(on)}), s)
        elif fn == 'cores':
            raise MgcError(s, 'cores() is reserved (level 3, intra-tile parallelism) and not implemented yet')
        elif on in ('dma',) or on in devices.HWPES or isinstance(self.env.get(fn), Kernel):
            self.env[name] = self.action(s.value, s)
        else:
            self.scalar(s, tgt, s.value, None)

    def alloc(self, s, name, kind):
        args = s.value.args
        view = self.ev(args[0]) if args else None
        if not isinstance(view, ir.View):
            raise MgcError(s, f'L1.{kind}() takes an L2 tensor view that fixes its shape')
        depth = 1
        for kw in s.value.keywords:
            if kw.arg == 'depth' and kind == 'multi_buffer':
                depth = const_of(self.expr(kw.value))
            else:
                raise MgcError(s, f'unknown argument {kw.arg}')
        if kind == 'multi_buffer' and (depth is None or depth < 2):
            raise MgcError(s, 'L1.multi_buffer() needs a constant depth >= 2')
        if not view.shape:
            raise MgcError(s, 'cannot allocate a scalar')
        op = self.add(ir.AllocOp.create(attributes={'sym': StringAttr(name), 'view': ir.ViewAttr(view),
                                                    'depth': IntAttr(depth)},
                                        result_types=[ir.BufType()]), s)
        op.res.name_hint = name
        self.env[name] = Buf(name, op.res, view, depth)

    # -- operations -----------------------------------------------------------
    def operand(self, n):
        """An L1 operand: (Buf, index expression)."""
        v = self.ev(n)
        if isinstance(v, Buf):
            if v.depth != 1:
                raise MgcError(n, f'index multi-buffer `{v.name}` to pick a slot')
            return v, Const(0)
        if isinstance(v, Slot) and isinstance(v.buf, Buf):
            return v.buf, v.index
        if isinstance(v, (Remote, Slot)):
            raise MgcError(n, 'neighbour buffers (`.on()`) can only be the destination of dma.store')
        raise MgcError(n, 'expected an L1 buffer or multi-buffer slot')

    def action(self, c, s):
        obj, fn = _call_parts(c)
        if fn is None:
            raise MgcError(c, 'expected a call')
        on = _name(obj)
        if fn == 'wait' and obj is not None and on not in ('dma',) and on not in devices.HWPES:
            ev = self.action(obj, s) if isinstance(obj, ast.Call) else self.ev(obj)
            if not isinstance(ev, ir.EvtRef):
                raise MgcError(c, '.wait() applies to events of DMA transfers and accelerator jobs')
            self.add(ir.WaitOp.create(attributes={'event': ir.EvtAttr(ev)}), s)
            return None
        if on == 'dma' and fn in ('load', 'store'):
            return self.dma(c, fn, s)
        if on in devices.HWPES:
            return self.hwpe(c, on, fn, s)
        k = self.env.get(fn) if on is None else None
        if isinstance(k, Kernel):
            return self.kernel_call(c, k, s)
        if on is None and fn == 'check':
            return self.check(c, s)
        if on is None and fn == 'sync':
            scope = _name(c.args[0]) if c.args else 'GLOBAL'
            if scope not in devices.SYNC:
                raise MgcError(c, f'sync scope must be one of {", ".join(devices.SYNC)}')
            self.add(ir.SyncOp.create(attributes={'scope': StringAttr(scope)}), s)
            return None
        if on is None and fn == 'rotate':
            v = _name(c.args[0]) if c.args else None
            if v is None:
                raise MgcError(c, 'rotate(var) names the variable that indexes the multi-buffers')
            self.add(ir.SelectOp.create(attributes={'var': StringAttr(v)}), s)
            return None
        raise MgcError(c, f'unknown operation `{ast.unparse(c.func)}`')

    def dma(self, c, fn, s):
        if len(c.args) != 2 or c.keywords:
            raise MgcError(c, f'dma.{fn}(dst, src)')
        dst, src = c.args
        attrs = {}
        operands = []
        if fn == 'load':
            buf, idx = self.operand(dst)
            view = self.ev(src)
            if not isinstance(view, ir.View):
                raise MgcError(src, 'the source of dma.load must be an L2 tensor view '
                               '(L1->L1 copies are not supported)')
            attrs['l2'] = ir.ViewAttr(view)
            d = 0
        else:
            buf, idx = self.operand(src)
            tv = self.ev(dst)
            d = 1
            if isinstance(tv, ir.View):
                attrs['l2'] = ir.ViewAttr(tv)
            elif isinstance(tv, Slot) and isinstance(tv.buf, Remote) or isinstance(tv, Remote):
                r = tv.buf if isinstance(tv, Slot) else tv
                ridx = tv.index if isinstance(tv, Slot) else Const(0)
                if r.buf is not buf:
                    raise MgcError(c, 'a neighbour store copies a buffer into the same buffer on the neighbour tile')
                if ridx.c() != idx.c():
                    raise MgcError(c, 'a neighbour store must target the same slot it reads')
                rop = self.add(ir.RemoteOp.create(operands=[buf.value],
                                                  attributes={'dy': ir.E(r.dy), 'dx': ir.E(r.dx)},
                                                  result_types=[ir.BufType()]), s)
                operands.append(rop.res)
            else:
                raise MgcError(dst, 'the destination of dma.store must be an L2 tensor view or a '
                               'neighbour buffer slot (`buf.on(dy, dx)[i]`)')
        ev = ir.EvtRef('dma', buf.name, dir=d)
        attrs.update({'dir': IntAttr(d), 'slot': ir.E(idx), 'event': ir.EvtAttr(ev)})
        self.add(ir.DmaOp.create(operands=[buf.value] + operands, attributes=attrs), s)
        return ir.EvtRef('dma', buf.name, dir=d)

    def _job_operands(self, c, s):
        bufs, slots = [], []
        for a in c.args:
            b, i = self.operand(a)
            bufs.append(b.value)
            slots.append(i)
        return bufs, slots

    def _event_kw(self, c, acc):
        kw = {k.arg: k.value for k in c.keywords}
        if 'after' in kw:
            raise MgcError(c, '`after=` is no longer needed: dependencies are inferred from the buffers')
        ev = None
        if 'event' in kw:
            ev = self.ev(kw.pop('event'))
            if not isinstance(ev, ir.EvtRef) or ev.kind != 'array':
                raise MgcError(c, '`event=` must be an element of an events() array')
        if kw:
            raise MgcError(c, f'unknown arguments: {", ".join(kw)}')
        return ev or ir.EvtRef('hwpe', acc)

    def hwpe(self, c, acc, fn, s):
        dev = devices.HWPES[acc]
        if fn in ('commit', 'start', 'commit_start'):
            job = next(iter(dev.jobs))
            self.add(ir.HwpeOp.create(attributes={'acc': StringAttr(acc), 'job': StringAttr(job),
                                                  'action': StringAttr(fn), 'slots': ArrayAttr([])}), s)
            return None
        if fn == 'enqueue':
            job = next(iter(dev.jobs))
            ev = self._event_kw(c, acc)
            bufs, slots = self._job_operands(c, s)
            self.add(ir.HwpeOp.create(operands=bufs, attributes={
                'acc': StringAttr(acc), 'job': StringAttr(job), 'action': StringAttr('enqueue'),
                'slots': ir.exprs(slots), 'event': ir.EvtAttr(ev)}), s)
            return ir.EvtRef(ev.kind, ev.name, index=ev.index)
        if fn in dev.jobs:
            ev = self._event_kw(c, acc)
            bufs, slots = self._job_operands(c, s)
            self.add(ir.JobOp.create(operands=bufs, attributes={
                'kernel': StringAttr(f'{acc}.{fn}'), 'slots': ir.exprs(slots), 'event': ir.EvtAttr(ev)}), s)
            return ir.EvtRef(ev.kind, ev.name, index=ev.index)
        raise MgcError(c, f'unknown {acc} operation `{fn}`')

    def kernel_call(self, c, k: Kernel, s):
        if c.keywords:
            raise MgcError(c, f'{k.sym}() takes its L1 operands positionally')
        if len(c.args) != len(k.operands):
            raise MgcError(c, f'{k.sym}() takes {len(k.operands)} L1 operands, got {len(c.args)}')
        bufs, slots = self._job_operands(c, s)
        self.add(ir.JobOp.create(operands=bufs, attributes={'kernel': StringAttr(k.sym),
                                                              'slots': ir.exprs(slots)}), s)
        return None

    def check(self, c, s):
        if len(c.args) != 2:
            raise MgcError(c, 'check(result_view, golden_view, tol=...)')
        a, b = self.ev(c.args[0]), self.ev(c.args[1])
        tol = 0
        for kw in c.keywords:
            if kw.arg == 'tol':
                tol = ast.literal_eval(kw.value)
        if not (isinstance(a, ir.View) and isinstance(b, ir.View)) or len(a.shape) != 2 or len(a.starts) != 2:
            raise MgcError(c, 'check() compares two 2-D L2 tensor views')
        ta, tb = self.env[a.tensor], self.env[b.tensor]
        if ta.dtype != tb.dtype:
            raise MgcError(c, 'check(): dtype mismatch')
        self.add(ir.CheckOp.create(attributes={'a': ir.ViewAttr(a), 'b': ir.ViewAttr(b), 'tol': IntAttr(tol)}), s)
        return None

    # -- loops ---------------------------------------------------------------
    def loop(self, s):
        _, fn = _call_parts(s.iter)
        if fn == 'tiles':
            raise MgcError(s, 'tiles() (level 1, spatial) must be the outermost loop of main')
        if fn == 'cores':
            raise MgcError(s, 'cores() is reserved (level 3, intra-tile parallelism) and not implemented yet')
        if fn not in ('range', 'pipeline') or not isinstance(s.target, ast.Name) or s.orelse:
            raise MgcError(s, 'time loops are `for v in range(n)` or `for v in pipeline(n)`')
        var = s.target.id
        args = s.iter.args
        saved = self.env.get(var)
        if fn == 'range':
            if len(args) not in (1, 2):
                raise MgcError(s, 'range(n) or range(a, b)')
            lo, hi = (Const(0), self.expr(args[0])) if len(args) == 1 else (self.expr(args[0]), self.expr(args[1]))
            ctype = 'int'

            def enter():
                self.env[var] = Sym(var, None, ctype)

            op = ir.ForOp.create(attributes={'var': StringAttr(var), 'lo': ir.E(lo), 'hi': ir.E(hi),
                                             'ctype': StringAttr(ctype)},
                                 regions=[self.region(s.body, enter=enter)])
        else:
            if len(args) != 1:
                raise MgcError(s, 'pipeline(n, ...) takes one trip count')
            attrs = {'var': StringAttr(var), 'n': ir.E(self.expr(args[0]))}
            for kw in s.iter.keywords:
                if kw.arg in ('skew', 'steps'):
                    attrs[kw.arg] = ir.E(self.expr(kw.value))
                elif kw.arg == 'sync':
                    sc = _name(kw.value)
                    if sc not in devices.SYNC:
                        raise MgcError(s, f'sync scope must be one of {", ".join(devices.SYNC)}')
                    attrs['sync'] = StringAttr(sc)
                elif kw.arg == 'time':
                    attrs['time'] = StringAttr(ast.literal_eval(kw.value))
                else:
                    raise MgcError(s, f'unknown pipeline() argument {kw.arg}')
            if ('skew' in attrs) != ('steps' in attrs):
                raise MgcError(s, 'pipeline(): `skew=` needs `steps=` (total time steps of the mesh)')

            def enter():
                self.env[var] = Sym(var, None, 'int')

            op = ir.PipelineOp.create(attributes=attrs, regions=[self.region(s.body, enter=enter)])
        self.add(op, s)
        if saved is None:
            self.env.pop(var, None)
        else:
            self.env[var] = saved


def parse(src, filename, header=None) -> ModuleOp:
    return Frontend(src, filename, header).run()
