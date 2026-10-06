# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""Compilation driver: front-end -> passes -> C."""

from __future__ import annotations

import io

from xdsl.context import Context
from xdsl.printer import Printer

from .emit_c import emit
from .frontend import parse
from .ir import MG
from .passes import PASSES

PASS_NAMES = ['frontend'] + [p.name for p in PASSES]


def print_ir(module) -> str:
    """Return the textual form of the IR module (what `--print-ir-after` shows)."""
    buf = io.StringIO()
    Printer(stream=buf).print_op(module)
    return buf.getvalue() + '\n'


def compile_module(src, filename, header=None, until=None, dump=None):
    """Run the front-end and the passes (up to and including `until`).
    `dump(name, text)` is called with the IR after each stage.

    Parameters: `src` is the .mgc text, `filename` is used in error messages,
    `header` is the test header path (optional), `until` is a stage name from
    `PASS_NAMES` at which to stop (default: run everything), `dump` an optional
    callback receiving `(stage_name, ir_text)`.

    Returns the IR module after the last executed stage. The module is verified
    after each stage. Raises `MgcError` for invalid programs.

    Example: `compile_module(src, 'a.mgc', until='pipeline')` returns the IR
    with the software pipelines already expanded into explicit guarded steps.
    """
    ctx = Context()
    ctx.load_dialect(MG)
    module = parse(src, filename, header)
    module.verify()
    if dump:
        dump('frontend', print_ir(module))
    if until == 'frontend':
        return module
    for p in PASSES:
        p.apply(ctx, module)
        module.verify()
        if dump:
            dump(p.name, print_ir(module))
        if until == p.name:
            break
    return module


def compile_source(src, filename, header=None) -> str:
    """Full compilation: `.mgc` text in, C source text out (unformatted).

    Same arguments as `compile_module`. This is what the tests use; `mgcc.py`
    additionally runs clang-format on the result.
    """
    return emit(compile_module(src, filename, header), filename)
