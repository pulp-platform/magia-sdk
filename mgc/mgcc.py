#!/usr/bin/env python3
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""mgcc - compile an mgc ("magic") test description into mglib C.

    mgcc.py tests/magia/mesh/mm_os_mgc/mm_os.mgc -o tests/magia/mesh/mm_os_mgc/src/test.c
    mgcc.py mm_os.mgc --print-ir-after=pipeline     # inspect the IR after a pass
    mgcc.py mm_os.mgc --print-ir-after-all

Needs xDSL: python3 -m venv mgc/.venv && mgc/.venv/bin/pip install -r mgc/requirements.txt
"""

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mgc.driver import PASS_NAMES, compile_module  # noqa: E402
from mgc.emit_c import emit  # noqa: E402
from mgc.errors import MgcError  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_clang_format():
    """Path of an executable `clang-format` (from `PATH`, else the repo's
    `llvm/install/bin`), or None if not found (output is then left unformatted)."""
    for p in (shutil.which('clang-format'), os.path.join(REPO, 'llvm', 'install', 'bin', 'clang-format')):
        if p and os.access(p, os.X_OK):
            return p
    return None


def main():
    """Command-line entry point: compile `source.mgc` to a C test.

    Reads the `.mgc` file, compiles it (see `mgc.driver`), formats it with
    clang-format when available and writes it to `-o FILE` or stdout. With
    `--print-ir-after=STAGE` it prints the IR at that stage and stops instead.
    Returns the process exit code: 0 on success, 1 after printing a compile error.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source', help='.mgc file')
    ap.add_argument('-o', '--output', help='output C file (default: stdout)')
    ap.add_argument('--header', help='test header with sizes and L2 data '
                    '(default: include/test.h next to the source)')
    ap.add_argument('--print-ir-after', choices=PASS_NAMES, help='print the IR after this stage and stop')
    ap.add_argument('--print-ir-after-all', action='store_true', help='print the IR after every stage to stderr')
    ap.add_argument('--no-format', action='store_true', help='do not run clang-format on the output')
    args = ap.parse_args()

    with open(args.source) as f:
        src = f.read()
    header = args.header or os.path.join(os.path.dirname(os.path.abspath(args.source)), 'include', 'test.h')
    try:
        if args.print_ir_after:
            def dump(name, text):
                if name == args.print_ir_after:
                    sys.stdout.write(text)
            compile_module(src, args.source, header, until=args.print_ir_after, dump=dump)
            return 0
        dump = (lambda name, text: sys.stderr.write(f'// ----- IR after {name}\n{text}')) \
            if args.print_ir_after_all else None
        c = emit(compile_module(src, args.source, header, dump=dump), args.source)
    except MgcError as e:
        print(e, file=sys.stderr)
        return 1

    cf = None if args.no_format else find_clang_format()
    if cf:
        # always the repo style, wherever the output goes
        style = os.path.join(REPO, '.clang-format')
        c = subprocess.run([cf, f'--style=file:{style}', '--assume-filename=test.c'], input=c,
                           capture_output=True, text=True, check=True).stdout
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(c)
    else:
        sys.stdout.write(c)
    return 0


if __name__ == '__main__':
    sys.exit(main())
