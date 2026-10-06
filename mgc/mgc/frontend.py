# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""Front-end: .mgc (Python syntax, parsed with `ast`, never executed) -> mg IR.

The front-end only resolves names and builds IR. Legality (DMA geometry,
shapes, event use) is checked by the passes, which report source lines.

Typical use (see `parse()`)::

    module = parse(open('mm_os.mgc').read(), 'mm_os.mgc', header='include/test.h')

Input (abridged `.mgc`)::

    M, N = sizes("M_SIZE", "N_SIZE")
    X = l2("x_inp", (M, N), fp16)

    @test("demo")
    def main():
        for y_id, x_id in tiles():
            x = L1.alloc(X[0:8, 0:8])
            dma.load(x, X[0:8, 0:8]).wait()

Output: an xDSL `ModuleOp` holding one `mg.tensor` per `l2()`, one `mg.define`
per `define()`, one `mg.kernel` per `l1_kernel()`, and a `mg.test` whose body
is a `mg.tiles` op containing the per-tile code (`mg.alloc`, `mg.dma`, ...).

How it works: module-level declarations and `main` are walked once; every Python
name is bound in `Frontend.env` to a compile-time value (`Sym`/`Expr` for
integers, `Tensor`, `Axis`, `Split`, `Buf`, `Kernel`, ...) and statements are
turned into IR ops.
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

# Tensor element types: name -> (C type used in the generated code, size in bytes).
DTYPES = {'fp16': ('uint16_t', 2)}
# Scalar variable types (`x: u8 = ...`): name -> (C type, largest value it can hold).
CTYPES = {'u8': ('uint8_t', 0xFF), 'u16': ('uint16_t', 0xFFFF), 'u32': ('uint32_t', 0xFFFFFFFF),
          'i32': ('int32_t', 0x7FFFFFFF)}
ROLES = ('in', 'out', 'inout')

# ----------------------------------------------------------------------------
# Compile-time values bound to Python names
# ----------------------------------------------------------------------------


@dataclass(eq=False)
class Tensor:
    """A tensor in L2 declared with `X = l2("x_inp", (M, N), fp16)`.

    `pyname` is the Python name (`X`), `cname` the C array holding the data
    (`x_inp`), `shape` the row-major dimensions as symbolic expressions
    (`[M, N]`) and `dtype` the element type (`fp16`).
    """
    pyname: str
    cname: str
    shape: List[Expr]
    dtype: str

    @property
    def esize(self):
        """Element size in bytes (2 for fp16)."""
        return DTYPES[self.dtype][1]

    @property
    def ctype(self):
        """C type of one element (`uint16_t` for fp16)."""
        return DTYPES[self.dtype][0]

    def stride(self, d) -> Expr:
        """Distance in elements between consecutive indices of dimension `d`.

        For shape `(M, N)`: `stride(0) == N`, `stride(1) == 1`.
        """
        return product(*self.shape[d + 1:])


@dataclass(eq=False)
class Axis:
    """A mesh coordinate bound by `for y_id, x_id in tiles():`.

    `var` is the run-time tile index (e.g. `y_id`), `axis` is `'y'` (mesh rows)
    or `'x'` (mesh columns). Its `.split(E)` method creates a `Split`.
    """
    var: Sym
    axis: str  # 'y' | 'x'


@dataclass(eq=False)
class Split:
    """The share of an extent that one tile owns, from `h = y_id.split(E)`.

    The extent `E` is cut in blocks of `h_max = ceil(E / MESH_Y_TILES)`; tile
    `y_id` owns `[h_max * y_id, h_max * y_id + h.size)`, with edge tiles clipped
    (e.g. E=10 on 4 rows: sizes 3, 3, 3, 1). `.start` and `.size` give the
    block origin and length; using `h` directly as a tensor index selects the
    whole block.
    """
    name: str
    axis: Axis
    extent: Expr

    @property
    def max(self):
        """Nominal block size `h_max` (same on all tiles, ignoring clipping)."""
        return Sym(self.name + '_max')

    @property
    def start(self):
        """First index owned by this tile: `h_max * y_id`."""
        return BinOp('*', self.max, self.axis.var)

    @property
    def size(self):
        """Number of indices owned by this tile (0 if the block is empty)."""
        return Sym(self.name, None, 'int32_t')


@dataclass(eq=False)
class Buf:
    """An L1 buffer from `L1.alloc(view)` (depth 1) or `L1.multi_buffer(view, depth=N)`.

    `value` is the SSA value of the `mg.alloc` op, `view` the L2 view that fixed
    its shape and `depth` the number of slots.
    """
    name: str
    value: object  # SSA value of the mg.alloc
    view: ir.View
    depth: int


@dataclass(eq=False)
class Remote:
    """`buf.on(dy, dx)`: the copy of `buf` in the L1 of tile `(y_id+dy, x_id+dx)`."""
    buf: Buf
    dy: Expr
    dx: Expr


@dataclass(eq=False)
class Slot:
    """One slot of a multi-buffer, `buf[index]` (`buf` may also be a `Remote`).

    `index` is a constant (`y[0]`) or a rotating index (`y[pt + 1]`).
    """
    buf: object  # Buf | Remote
    index: Expr


@dataclass(eq=False)
class EvArray:
    """An array of `depth` accelerator events from `ev = redmule.events(n)`.

    `kind` is the accelerator name (`redmule`).
    """
    name: str
    depth: int
    kind: str


@dataclass(eq=False)
class Kernel:
    """A software kernel from `k = l1_kernel("c_fn", operands=("dst:out", ...), ...)`.

    `sym` is the Python name, `cname` the C function, `operands` the list of
    `(name, role)` with role in `in`/`out`/`inout`, `params` the extra integer
    arguments (`"src.size"`) and `executor` who runs it (`CORE`).
    """
    sym: str
    cname: str
    operands: List[tuple]  # (name, role)
    params: List[str]
    executor: str


# ----------------------------------------------------------------------------


def parse_header(path):
    """Read the test header (e.g. `include/test.h`) that accompanies a `.mgc` file.

    Returns `(defs, text)`: `defs` maps every simple numeric `#define NAME 123`
    (or `(123)`) to its integer value, `text` is the raw file content (used to
    check that L2 array names exist). Missing/None path -> `({}, '')`, which
    disables the header checks.

    Example: a header with `#define M_SIZE 96` gives `defs == {'M_SIZE': 96}`.
    """
    defs, text = {}, ''
    if path and os.path.exists(path):
        with open(path) as f:
            text = f.read()
        for m in re.finditer(r'^\s*#define\s+(\w+)\s+\(?\s*(\d+)\s*\)?\s*$', text, re.M):
            defs[m.group(1)] = int(m.group(2))
    return defs, text


def _comments(src):
    """Map line number -> text of every standalone `# comment` line in `src`.

    Trailing comments (after code) are ignored. `"  # hi"` on line 4 gives `{4: 'hi'}`.
    """
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
    """Translates one `.mgc` source into an `mg` IR module.

    Walks the Python AST and builds IR ops block by block. Keeps:

    - `env`: every Python name seen so far -> its compile-time value
      (`Sym`, `Tensor`, `Axis`, `Split`, `Buf`, `Kernel`, `EvArray`, event refs);
    - `hdefs`/`htext`: macros and text of the test header, used for checks;
    - `comments`/`cursor`: standalone `#` comments, re-emitted as IR comment
      ops in source order.

    Use `parse()` rather than instantiating this directly; `run()` does the work.
    """

    def __init__(self, src, filename, header=None):
        """`src`: .mgc text; `filename`: used in error messages; `header`: path of
        the test header (optional; enables `sizes()`/`l2()` checks)."""
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
        """Append `op` to the block being built, tagging it with the source line of `node`."""
        if node is not None and hasattr(node, 'lineno'):
            op.lineno = node.lineno
        self.blocks[-1].add_op(op)
        return op

    def region(self, stmts, top=False, enter=None):
        """Translate `stmts` into a new IR region (the body of a loop or `if` branch).

        `top`: the statements are the top of the `tiles()` body (tile splits are
        only allowed there). `enter`: callback run first, to bind the loop variable.
        """
        blk = Block()
        self.blocks.append(blk)
        if enter:
            enter()
        self.block(stmts, top)
        self.blocks.pop()
        return Region(blk)

    # -- module level --------------------------------------------------------
    def run(self) -> ModuleOp:
        """Compile the whole file; returns the `ModuleOp`.

        Module level accepts imports, a docstring, declarations (`sizes`, `l2`,
        `define`, `l1_kernel`) and a single `@test(...)`-decorated `main` whose
        body is exactly one `for y_id, x_id in tiles():` loop, optionally
        preceded by a docstring (stored as the test's `doc`).

        Result: `[mg.tensor/mg.define/mg.kernel ..., mg.test{ mg.tiles{ ... } }]`.
        Raises `MgcError` (with the source line) on anything unsupported.
        """
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
        """Handle one module-level assignment. Supported forms:

        - `M, N = sizes("M_SIZE", "N_SIZE")`: bind names to header macros;
        - `X = l2("x_inp", (M, N), fp16)`: declare an L2 tensor (emits `mg.tensor`);
        - `ITERS = define(4)`: new `#define` owned by the .mgc (emits `mg.define`);
        - `k = l1_kernel(...)`: see `l1_kernel()`.
        """
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
        """Declare a software kernel, e.g.

            scale = l1_kernel("scale_fp16", operands=("dst:out", "src:in"),
                              params=("src.size",), executor=CORE)

        Later `scale(d, s)` in the tile loop becomes the C call
        `scale_fp16(addr_d, addr_s, size_of_s)`. Operands are L1 buffers with a
        role (used to order operations); params are integer arguments derived
        from operand shapes. Emits a `mg.kernel` op and binds `sym` to a `Kernel`.
        """
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
        """Evaluate expression node `n` at compile time. The result depends on `n`:

        - integer expression -> `Expr` (`y_id * 2 + 1`, `M // 4`, `MESH_X_TILES`);
        - bare name -> whatever it is bound to (`Tensor`, `Buf`, `Split`, ...);
          tile axes (`y_id`) evaluate to their run-time variable;
        - `h.start` / `h.size` -> `Expr` for a tile split;
        - `b.on(dy, dx)` -> `Remote`; `b[i]` -> `Slot`; `ev[i]` -> event ref;
        - `X[a:b, h]` -> `ir.View` (see `view()`);
        - comparisons / `and` / `or` / `not` -> condition `Expr`.

        Use `expr()` when an integer is required.
        """
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
        """Like `ev()`, but the result must be an integer `Expr` (else `MgcError`)."""
        v = self.ev(n)
        if not isinstance(v, Expr):
            raise MgcError(n, f'`{ast.unparse(n)}` is not an integer expression')
        return v

    def cond(self, n) -> Expr:
        """Evaluate a condition (`if` test): comparisons, chains (`0 <= a < b`),
        `and`/`or`/`not`. Falls back to `expr()` for plain integers."""
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
        """Build the `ir.View` for `T[idx...]` on an L2 tensor.

        Per dimension the index is a unit-step slice `a:b` (or `:`), a tile
        split (selects the tile's block), or an integer (selects one row and
        drops the dimension). Missing trailing indices mean `:`.

        For `X` of shape (M, N): `X[2:6, h]` -> starts `[2, h.start]`, sizes
        `[4, h.size]`, both dims kept; `X[3, :]` -> starts `[3, 0]`, sizes
        `[1, N]`, first dim dropped (view shape `[N]`).
        """
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
        """Emit IR comment ops for standalone `#` lines found between the last
        processed statement and `s`, so they reappear in the generated C."""
        if not hasattr(s, 'lineno'):
            return
        for ln in sorted(self.comments):
            if self.cursor < ln < s.lineno:
                self.add(ir.CommentOp.create(attributes={'text': StringAttr(self.comments[ln]),
                                                         'style': StringAttr('line')}))
        self.cursor = max(self.cursor, s.lineno)

    def block(self, stmts, top=False):
        """Translate a list of statements into the current block.
        `top` allows tile splits (only legal at the top of the `tiles()` body)."""
        for s in stmts:
            if self._is_split(s) and not top:
                raise MgcError(s, 'tile splits must be at the top of the tiles() body')
            self.stmt(s)

    @staticmethod
    def _is_split(s):
        return isinstance(s, ast.Assign) and _call_parts(s.value)[1] == 'split'

    def stmt(self, s):
        """Translate one statement inside the tile loop.

        Dispatches on the kind: docstring/`comment()` -> comment op;
        `x: u8 = e` / `x = e` / `x += e` -> scalar (`scalar()`); other
        assignments -> `assign()`; bare calls (`dma.load(...)`, `sync(ROW)`,
        `check(...)`) -> `action()`; `for` -> `loop()`; `if` -> `mg.if`;
        `continue`; `pass`. Anything else is an `MgcError`.
        """
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
        """Integer variable assignment.

        `t_size: u8 = M // 4` declares a new variable (C type from `ann`,
        range-checked when the value is a compile-time constant);
        `n = e` declares a `uint32_t` the first time and updates it later;
        `n += 1` is rewritten to `n = n + 1`. Emits `mg.scalar` (`decl` = 1 for
        a declaration, 0 for an update).
        """
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
        """Plain `name = <call or expr>` inside the tile loop. The right side picks the meaning:

        - `y_id.split(E)` -> tile split (`mg.split`);
        - `L1.alloc(view)` / `L1.multi_buffer(view, depth=N)` -> L1 buffer (`alloc()`);
        - `redmule.events(n)` -> event array (`mg.events`);
        - `dma.load/store(...)`, `redmule.gemm(...)`, `kernel(...)` -> the name is
          bound to the returned event, so `e.wait()` works later;
        - anything else -> integer scalar (`scalar()`).
        """
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
        """`b = L1.alloc(X[r, c])` or `b = L1.multi_buffer(X[r, c], depth=3)`.

        The view argument only provides the buffer's shape (and transfer
        geometry); no data moves. Emits `mg.alloc` and binds `name` to a `Buf`
        with `depth` slots (1 for `alloc`, >= 2 for `multi_buffer`).
        """
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
        """Translate a call statement; returns an event reference (or None).

        Handled calls: `e.wait()` / `dma.load(...).wait()`, `dma.load/store`,
        accelerator calls (`redmule.gemm`, `.enqueue`, `.commit`, ...), kernel
        calls, `check(a, b, tol=...)`, `sync(SCOPE)`, `rotate(var)`.
        The returned event lets callers chain `.wait()` or store it in a name.
        """
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
        """`dma.load(dst_slot, l2_view)` / `dma.store(dst, src_slot)`.

        - `dma.load(x[0], X[0:8, :])`: L2 -> L1;
        - `dma.store(Y[0:8, :], y[0])`: L1 -> L2;
        - `dma.store(y.on(1, 0)[0], y[0])`: L1 -> same slot of the same buffer
          on the tile below (neighbour store).

        Emits `mg.dma` and returns the DMA event of that buffer/direction, to
        be waited with `.wait()`. Geometry is checked later by `legalize-dma`.
        """
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
        """Split the positional args of a job call into parallel lists:
        buffer SSA values and slot-index expressions (`x[pt]` -> `x`, `pt`)."""
        bufs, slots = [], []
        for a in c.args:
            b, i = self.operand(a)
            bufs.append(b.value)
            slots.append(i)
        return bufs, slots

    def _event_kw(self, c, acc):
        """Event on which an accelerator job completes: the `event=ev[k]` argument
        if given, else the accelerator's default event."""
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
        """Accelerator calls on `acc` (e.g. `redmule`):

        - `redmule.gemm(x, w, y)` (a job named in `devices.HWPES`): one-shot job,
          computes `y += x @ w`; returns its event;
        - `redmule.enqueue(x, w, y, event=ev[k])`: program a job without starting it;
        - `redmule.commit()` / `.start()` / `.commit_start()`: explicit queue control.
        """
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
        """Call of an `l1_kernel`, e.g. `scale(dst[0], src[0])`: positional L1
        operands in the declared order. Emits a synchronous `mg.job`."""
        if c.keywords:
            raise MgcError(c, f'{k.sym}() takes its L1 operands positionally')
        if len(c.args) != len(k.operands):
            raise MgcError(c, f'{k.sym}() takes {len(k.operands)} L1 operands, got {len(c.args)}')
        bufs, slots = self._job_operands(c, s)
        self.add(ir.JobOp.create(operands=bufs, attributes={'kernel': StringAttr(k.sym),
                                                              'slots': ir.exprs(slots)}), s)
        return None

    def check(self, c, s):
        """`check(Y[:, tw], Z[:, tw], tol=0x11)`: compare a computed 2-D L2 region
        against golden data, element by element, with absolute tolerance `tol`
        (default 0). The generated test returns the number of mismatches."""
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
        """Time loops inside the tile loop.

        - `for i in range(n)` / `range(a, b)` -> plain loop (`mg.for`);
        - `for pt in pipeline(n, skew=S, steps=T, sync=SCOPE)` -> software
          pipeline over `n` iterations (`mg.pipeline`); `skew` and `steps` must
          be given together (this tile starts `S` steps late out of `T` total),
          `sync` adds a barrier per step, `time` is a label.

        The loop variable is only in scope inside the body. `tiles()` and
        `cores()` are rejected here.
        """
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
    """Entry point of the front-end: `.mgc` source text -> `mg` IR module.

    `src`: contents of the `.mgc` file; `filename`: shown in error messages;
    `header`: optional path of the test header (e.g. `include/test.h`) used to
    check `sizes()` macros and `l2()` array names.

    >>> module = parse(open('mm_os.mgc').read(), 'mm_os.mgc', 'include/test.h')

    Raises `MgcError` on invalid input.
    """
    return Frontend(src, filename, header).run()
