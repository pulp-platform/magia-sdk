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

        operatorRepresentation['offset'] = 0
        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaMatMulFP16Spatz("""
// Magia MatMul FP16 Spatz (Name: ${nodeName}, Op: {nodeOp})
MAGIA_matmul_fp16_spatz(${A}, ${B}, ${data_out}, ${M}, ${N}, ${O}, ${batch});
""")
