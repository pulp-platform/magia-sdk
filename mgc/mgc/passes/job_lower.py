# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""job-lower: jobs -> executor-specific operations.

* `mg.job` on an HWPE ("redmule.gemm")  -> `mg.hwpe` oneshot (enqueue+commit+start).
  (Pipelined jobs were already split into enqueue / commit / start by the
  pipeline pass, which owns that scheduling decision.)
* `mg.job` on a software L1 kernel        -> `mg.call` (synchronous, CV32 core).

For every job (and HWPE enqueue) the scalar parameters are derived from the
operand shapes and the constraints are checked, both from the descriptors in
devices.py / the kernel declaration: a new accelerator or kernel needs no
change here.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import ArrayAttr, StringAttr
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter

from .. import devices
from .. import ir
from ..errors import MgcError
from ..expr import equal
from .common import eval_shape_expr, kernels, operand_shapes, shape, alloc_of, tensors


def hwpe_params(op, acc, job, ts):
    """Check a HWPE job call against its descriptor and compute its scalar arguments.

    For `redmule.gemm(x, w, y)` with x: (M, K), w: (K, N), y: (M, N) it checks the
    operand count/rank and that the shapes agree, and returns the expressions
    `[M, K, N]` passed to `mg_redmule_gemm`. Raises `MgcError` on mismatch.
    `ts` is `tensors(module)`.
    """
    spec = devices.HWPES[acc].jobs[job]
    names = [n for n, _, _ in spec.operands]
    if len(op.bufs) != len(names):
        raise MgcError(op, f'{acc}.{job} takes {len(names)} L1 operands ({", ".join(names)})')
    for (n, _, rank), v in zip(spec.operands, op.bufs):
        r = len(shape(alloc_of(v)))
        if r != rank:
            raise MgcError(op, f'{acc}.{job}: operand {n} must be {rank}-D, `{alloc_of(v).sym.data}` is {r}-D')
    shp = operand_shapes(op, names, ts)
    for a, b in spec.constraints:
        if not equal(eval_shape_expr(a, shp, op), eval_shape_expr(b, shp, op)):
            desc = ' '.join(f'{n}{[e.c() for e in shp[n].shape]}' for n in names)
            raise MgcError(op, f'{acc}.{job} shapes do not match ({a} != {b}): {desc}')
    return [eval_shape_expr(e, shp, op) for _, e in spec.params]


@dataclass(frozen=True)
class JobLower(ModulePass):
    """Turns every high-level `mg.job` into a concrete op:
    `redmule.gemm(x, w, y)` -> `mg.hwpe` one-shot with params `[M, K, N]`;
    `scale(dst, src)` -> `mg.call` with params from the kernel's `params=`.
    """
    name = 'job-lower'

    def apply(self, ctx, module):
        ts = tensors(module)
        kdefs = kernels(module)
        for op in list(module.walk()):
            if isinstance(op, ir.HwpeOp) and op.action.data == 'enqueue':
                op.attributes['params'] = ir.exprs(hwpe_params(op, op.acc.data, op.job.data, ts))
            elif isinstance(op, ir.JobOp):
                k = op.kernel.data
                if '.' in k:
                    acc, job = k.split('.')
                    params = hwpe_params(op, acc, job, ts)
                    new = ir.HwpeOp.create(operands=list(op.bufs), attributes={
                        'acc': StringAttr(acc), 'job': StringAttr(job), 'action': StringAttr('oneshot'),
                        'slots': op.slots, 'event': op.event, 'params': ir.exprs(params)})
                else:
                    kd = kdefs[k]
                    names = [o.data.partition(':')[0] for o in kd.operands_.data]
                    shp = operand_shapes(op, names, ts)
                    params = [eval_shape_expr(p.data, shp, op) for p in kd.params.data]
                    new = ir.CallOp.create(operands=list(op.bufs), attributes={
                        'kernel': StringAttr(k), 'slots': op.slots, 'params': ir.exprs(params)})
                new.lineno = getattr(op, 'lineno', None)
                Rewriter.replace_op(op, new)
