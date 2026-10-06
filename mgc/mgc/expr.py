# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""Tiny symbolic integer expressions.

Expressions keep the tree shape the user wrote (so the emitted C reads like
the source) and can be normalized to a polynomial over atoms when the compiler
needs to reason about them (slice sizes, address differences, equality).
Division and modulo are not polynomial: they become opaque atoms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

Poly = Dict[Tuple[str, ...], int]

_PREC = {'+': 1, '-': 1, '*': 2, '/': 2, '%': 2}


class Expr:
    """Base class of integer/boolean expression trees.

    Python operators build trees: `Sym('N') * 2 + 1` is `N * 2 + 1`. Every node
    can print itself as C (`c()`), report a static value when all leaves are
    known (`value()`), list the symbols it uses (`syms()`), substitute symbols
    (`subst()`) and reduce to a polynomial (`poly()`) for comparisons.
    """

    # -- construction ------------------------------------------------------
    def __add__(self, o):
        return add(self, lift(o))

    def __radd__(self, o):
        return add(lift(o), self)

    def __sub__(self, o):
        return sub(self, lift(o))

    def __rsub__(self, o):
        return sub(lift(o), self)

    def __mul__(self, o):
        return mul(self, lift(o))

    def __rmul__(self, o):
        return mul(lift(o), self)

    # -- analysis ----------------------------------------------------------
    def poly(self) -> Poly:
        """Normal form as a polynomial: `{('N',): 2, (): 1}` means `2*N + 1`.
        Keys are sorted tuples of atoms, `()` is the constant term."""
        raise NotImplementedError

    def value(self) -> Optional[int]:
        """Static value, if every leaf has one."""
        raise NotImplementedError

    def c(self) -> str:
        """The expression as C source text, e.g. `'y_id * 2 + 1'`."""
        raise NotImplementedError

    def prec(self) -> int:
        """Operator precedence used to decide where parentheses are needed (3 = atom)."""
        return 3

    def syms(self) -> set:
        return set()

    def subst(self, m: Dict[str, 'Expr']) -> 'Expr':
        """Replace symbols by expressions (folding constants on the way)."""
        return self

    def __repr__(self):
        return self.c()


@dataclass(eq=False, repr=False)
class Const(Expr):
    """An integer literal: `Const(4).c() == '4'`."""
    v: int

    def poly(self):
        return {(): self.v} if self.v else {}

    def value(self):
        return self.v

    def c(self):
        return str(self.v)


@dataclass(eq=False, repr=False)
class Sym(Expr):
    """A named C entity: a macro, a C variable or a loop counter."""
    name: str
    static: Optional[int] = None  # compile-time value, when known
    ctype: str = 'uint32_t'

    def poly(self):
        return {(self.name,): 1}

    def value(self):
        return self.static

    def c(self):
        return self.name

    def syms(self):
        return {self.name}

    def subst(self, m):
        return m.get(self.name, self)


@dataclass(eq=False, repr=False)
class Raw(Expr):
    """Verbatim C text (e.g. a cast). Opaque to the analysis."""
    text: str

    def poly(self):
        return {(self.text,): 1}

    def value(self):
        return None

    def c(self):
        return self.text


@dataclass(eq=False, repr=False)
class Paren(Expr):
    """Explicit parentheses (printing only): `(a * b * 2)` as in hand-written C."""
    a: Expr

    def poly(self):
        return self.a.poly()

    def value(self):
        return self.a.value()

    def c(self):
        return '(' + self.a.c() + ')'

    def syms(self):
        return self.a.syms()

    def subst(self, m):
        return Paren(self.a.subst(m))


@dataclass(eq=False, repr=False)
class BinOp(Expr):
    """Binary arithmetic `a op b` with op in `+ - * / %` (`/` is integer division).

    Prints with the minimum parentheses that keep C semantics:
    `BinOp('*', a+b, c)` -> `(a + b) * c`.
    """
    op: str
    a: Expr
    b: Expr

    def prec(self):
        return _PREC[self.op]

    def syms(self):
        return self.a.syms() | self.b.syms()

    def subst(self, m):
        a, b = self.a.subst(m), self.b.subst(m)
        if self.op == '+':
            return add(a, b)
        if self.op == '-':
            return sub(a, b)
        if self.op == '*':
            return mul(a, b)
        v = BinOp(self.op, a, b)
        return Const(v.value()) if v.value() is not None and not v.syms() else v

    def poly(self):
        if self.op == '+':
            return _padd(self.a.poly(), self.b.poly(), 1)
        if self.op == '-':
            return _padd(self.a.poly(), self.b.poly(), -1)
        if self.op == '*':
            return _pmul(self.a.poly(), self.b.poly())
        v = self.value()
        if v is not None and _is_const_poly(self.a.poly()) and _is_const_poly(self.b.poly()):
            return {(): v} if v else {}
        return {('(' + self.c() + ')',): 1}

    def value(self):
        a, b = self.a.value(), self.b.value()
        if a is None or b is None:
            return None
        if self.op == '+':
            return a + b
        if self.op == '-':
            return a - b
        if self.op == '*':
            return a * b
        if b == 0:
            return None
        return a // b if self.op == '/' else a % b

    def c(self):
        p = self.prec()
        sa = self.a.c()
        if self.a.prec() < p:
            sa = '(' + sa + ')'
        sb = self.b.c()
        # right operand: parenthesize equal precedence unless associative
        # (integer a * (b / c) != a * b / c)
        if self.b.prec() < p or (self.b.prec() == p and (self.op in '-/%' or self.b.op in '/%')):
            sb = '(' + sb + ')'
        return f'{sa} {self.op} {sb}'


def _is_const_poly(p):
    return all(k == () for k in p)


def _padd(p: Poly, q: Poly, s: int) -> Poly:
    r = dict(p)
    for k, v in q.items():
        r[k] = r.get(k, 0) + s * v
        if r[k] == 0:
            del r[k]
    return r


def _pmul(p: Poly, q: Poly) -> Poly:
    r: Poly = {}
    for k1, v1 in p.items():
        for k2, v2 in q.items():
            k = tuple(sorted(k1 + k2))
            r[k] = r.get(k, 0) + v1 * v2
            if r[k] == 0:
                del r[k]
    return r


def lift(x) -> Expr:
    """Turn a Python int into `Const`; pass an `Expr` through unchanged."""
    if isinstance(x, Expr):
        return x
    if isinstance(x, int):
        return Const(x)
    raise TypeError(f'cannot use {x!r} in an integer expression')


def is_zero(e: Expr) -> bool:
    """True if `e` is identically 0 as a polynomial (`N - N` is zero)."""
    return not e.poly()


def is_one(e: Expr) -> bool:
    """True if `e` is identically 1."""
    return e.poly() == {(): 1}


def equal(a: Expr, b: Expr) -> bool:
    """Mathematical equality: `N * 2 + 4` equals `4 + 2 * N`. Division/modulo
    are opaque, so `N / 2` only equals the same textual `N / 2`."""
    return is_zero(sub(a, b, fold=False))


def const_of(e: Expr) -> Optional[int]:  # e.g. `N - N + 3` -> 3, `N + 1` -> None
    """The integer value of `e` if its polynomial is a constant."""
    p = e.poly()
    if not p:
        return 0
    if list(p) == [()]:
        return p[()]
    return None


# Light folding only: keep the user's expression shape.
def add(a: Expr, b: Expr) -> Expr:
    """`a + b`, dropping zeros and folding trailing constants:
    `(x + 1) + 2` -> `x + 3`; `x + (-2)` -> `x - 2`."""
    if is_zero(b) and const_of(b) == 0:
        return a
    if is_zero(a) and const_of(a) == 0:
        return b
    if isinstance(a, Const) and isinstance(b, Const):
        return Const(a.v + b.v)
    # (x + c1) + c2 -> x + (c1 + c2)
    if isinstance(b, Const) and isinstance(a, BinOp) and a.op == '+' and isinstance(a.b, Const):
        return add(a.a, Const(a.b.v + b.v))
    if isinstance(b, Const) and b.v < 0:
        return BinOp('-', a, Const(-b.v))
    return BinOp('+', a, b)


def sub(a: Expr, b: Expr, fold=True) -> Expr:
    """`a - b`. With `fold=False` no simplification is done (used so that
    `equal()` can compare polynomials)."""
    if fold and const_of(b) == 0:
        return a
    if fold and isinstance(a, Const) and isinstance(b, Const):
        return Const(a.v - b.v)
    return BinOp('-', a, b)


def mul(a: Expr, b: Expr) -> Expr:
    """`a * b`, simplifying multiplication by 0 and 1 and constant products."""
    if const_of(a) == 0 or const_of(b) == 0:
        return Const(0)
    if const_of(b) == 1:
        return a
    if const_of(a) == 1:
        return b
    if isinstance(a, Const) and isinstance(b, Const):
        return Const(a.v * b.v)
    return BinOp('*', a, b)


def product(*xs: Expr) -> Expr:
    """Product of any number of factors; `product()` is `Const(1)`.
    Used for strides: `product(N, K)` -> `N * K`."""
    r: Expr = Const(1)
    for x in xs:
        r = mul(r, lift(x))
    return r


# ----------------------------------------------------------------------------
# Conditions (if / guard predicates). They print like C and evaluate statically
# when every leaf is known.
# ----------------------------------------------------------------------------

_CMP = {'<': lambda a, b: a < b, '<=': lambda a, b: a <= b, '>': lambda a, b: a > b,
        '>=': lambda a, b: a >= b, '==': lambda a, b: a == b, '!=': lambda a, b: a != b}


def _operand(e: Expr) -> str:
    return e.c() if e.prec() == 3 else '(' + e.c() + ')'


@dataclass(eq=False, repr=False)
class Cmp(Expr):
    """Comparison `a op b` (`< <= > >= == !=`), printed as C, e.g. `y_id == 0`."""
    op: str
    a: Expr
    b: Expr

    def prec(self):
        return 0

    def poly(self):
        return {('(' + self.c() + ')',): 1}

    def value(self):
        a, b = self.a.value(), self.b.value()
        return None if a is None or b is None else int(_CMP[self.op](a, b))

    def c(self):
        return f'{_operand(self.a)} {self.op} {_operand(self.b)}'

    def syms(self):
        return self.a.syms() | self.b.syms()

    def subst(self, m):
        return Cmp(self.op, self.a.subst(m), self.b.subst(m))


@dataclass(eq=False, repr=False)
class Logic(Expr):
    """`&&` / `||` of the conditions in `xs`: `Logic('&&', [a, b])` -> `a && b`."""
    op: str  # '&&' or '||'
    xs: list

    def prec(self):
        return -1 if self.op == '||' else -0.5

    def poly(self):
        return {('(' + self.c() + ')',): 1}

    def value(self):
        vs = [x.value() for x in self.xs]
        if self.op == '&&':
            if any(v == 0 for v in vs):
                return 0
            return 1 if all(v is not None for v in vs) else None
        if any(v not in (0, None) for v in vs):
            return 1
        return 0 if all(v is not None for v in vs) else None

    def c(self):
        parts = []
        for x in self.xs:
            parts.append(x.c() if x.prec() > self.prec() else '(' + x.c() + ')')
        return f' {self.op} '.join(parts)

    def syms(self):
        r = set()
        for x in self.xs:
            r |= x.syms()
        return r

    def subst(self, m):
        return Logic(self.op, [x.subst(m) for x in self.xs])


@dataclass(eq=False, repr=False)
class Not(Expr):
    """Logical negation: `Not(y_id == 0)` -> `!(y_id == 0)`."""
    a: Expr

    def poly(self):
        return {('(' + self.c() + ')',): 1}

    def value(self):
        v = self.a.value()
        return None if v is None else int(not v)

    def c(self):
        return '!' + _operand(self.a) if self.a.prec() == 3 else f'!({self.a.c()})'

    def syms(self):
        return self.a.syms()

    def subst(self, m):
        return Not(self.a.subst(m))


def same(a: Expr, b: Expr) -> bool:
    """Structural equality (as printed)."""
    return a.c() == b.c()
