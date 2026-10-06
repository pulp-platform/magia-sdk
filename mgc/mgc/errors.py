# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>


class MgcError(Exception):
    """A user-facing compile error, located at an AST node when possible."""

    filename = '<mgc>'

    def __init__(self, node, msg):
        self.lineno = getattr(node, 'lineno', None)
        self.msg = msg
        loc = f'{MgcError.filename}:{self.lineno}' if self.lineno else MgcError.filename
        super().__init__(f'{loc}: error: {msg}')
