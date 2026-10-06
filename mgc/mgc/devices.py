# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""Hardware descriptions: the only place passes learn what the hardware can do.

* Adding an iDMA mode (3-D, 4-D, strided L1, ...) means editing `IDMA`.
* Adding an HWPE accelerator means adding an `Hwpe` entry (and its mglib
  wrapper); no pass changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

# ----------------------------------------------------------------------------
# iDMA
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class IdmaCaps:
    max_rank: int  # dimensions of one transfer (1 = contiguous, 2 = rows x contiguous, ...)
    l1_strided: bool  # can the L1 (OBI) side be strided? (mglib: no)
    functions: Dict[int, str]  # rank -> mglib function


IDMA = IdmaCaps(max_rank=2, l1_strided=False,
                functions={1: 'mg_idma_memcpy_1d', 2: 'mg_idma_memcpy_2d'})

# ----------------------------------------------------------------------------
# HWPE accelerators (acquire/enqueue -> commit -> start -> event)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class HwpeJob:
    operands: Tuple[Tuple[str, str, int], ...]  # (name, role in|out|inout, rank)
    params: Tuple[Tuple[str, str], ...]  # (C type, expression over operand shapes)
    constraints: Tuple[Tuple[str, str], ...]  # pairs of shape expressions that must be equal


@dataclass(frozen=True)
class Hwpe:
    name: str
    ctrl: str  # controller variable in the test
    queue_depth: int  # hardware job queue
    jobs: Dict[str, HwpeJob]
    fn: Dict[str, str] = field(default_factory=dict)  # action -> mglib function ({job} expands)


HWPES: Dict[str, Hwpe] = {
    'redmule':
    Hwpe(name='redmule',
         ctrl='&redmule_ctrl',
         queue_depth=2,
         jobs={
             # y += x @ w
             'gemm':
             HwpeJob(operands=(('x', 'in', 2), ('w', 'in', 2), ('y', 'inout', 2)),
                     params=(('uint16_t', 'x.shape[0]'), ('uint16_t', 'x.shape[1]'),
                             ('uint16_t', 'w.shape[1]')),
                     constraints=(('x.shape[1]', 'w.shape[0]'), ('x.shape[0]', 'y.shape[0]'),
                                  ('w.shape[1]', 'y.shape[1]'))),
         },
         fn={
             'enqueue': 'mg_redmule_{job}_enqueue',
             'oneshot': 'mg_redmule_{job}',
             'commit': 'mg_redmule_{job}_commit',
             'start': 'mg_redmule_{job}_start',
             'commit_start': 'mg_redmule_{job}_commit_start',
             'wait': 'mg_redmule_wait',
         }),
}

# ----------------------------------------------------------------------------
# Software executors for L1 kernels
# ----------------------------------------------------------------------------

# CORE: the tile's CV32 control core, called synchronously. (Spatz / PULP
# offload will be added as asynchronous executors once their mglib layer exists.)
EXECUTORS = ('CORE',)

# Synchronization scopes -> fsync calls
SYNC = {'GLOBAL': 'fsync_sync_global', 'ROW': 'fsync_sync_row', 'COL': 'fsync_sync_col'}
