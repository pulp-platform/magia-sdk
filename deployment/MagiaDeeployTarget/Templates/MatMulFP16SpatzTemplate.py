# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaMatMulFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt.lookup(operatorRepresentation['A'])
        ctxt.lookup(operatorRepresentation['B'])
        ctxt.lookup(operatorRepresentation['data_out'])

        # A_batched/B_batchetd means operand should be broadcasted
        operatorRepresentation['a_batched'] = int(bool(operatorRepresentation['A_batched']))
        operatorRepresentation['b_batched'] = int(bool(operatorRepresentation['B_batched']))

        operatorRepresentation['offset'] = 0
        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaMatMulFP16Spatz("""
// Magia MatMul FP16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
#ifdef ENABLE_NODE_LOGS
printf("[CV32 (%d)] Running node: ${nodeName} (${nodeOp})\\n", get_hartid());
#endif
MAGIA_matmul_fp16_spatz(${A}, ${B}, ${data_out}, ${M}, ${N}, ${O}, ${batch}, ${a_batched}, ${b_batched});
""")
