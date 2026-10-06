# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>


class MgcError(Exception):
    """A user-facing compile error, located at an AST node when possible.

    `MgcError(node, "msg")` is rendered as `file.mgc:12: error: msg`, taking the
    line from `node.lineno` (or just `file.mgc: error: msg` if `node` is None or
    has no line). The file name is the class attribute `filename`, set once by
    the front-end.
    """

    filename = '<mgc>'

    def __init__(self, node, msg):
        """`node`: AST node or IR op carrying `lineno` (or None); `msg`: text."""
        self.lineno = getattr(node, 'lineno', None)
        self.msg = msg
        loc = f'{MgcError.filename}:{self.lineno}' if self.lineno else MgcError.filename
        super().__init__(f'{loc}: error: {msg}')
