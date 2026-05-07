# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _OnnxAddTemplate(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt.lookup(operatorRepresentation['data_in_1'])
        ctxt.lookup(operatorRepresentation['data_in_2'])
        ctxt.lookup(operatorRepresentation['data_out'])
        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _OnnxAddTemplate("""
// Magia Onnx Add (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_onnx_add(${data_in_1}, ${data_in_2}, ${data_out}, ${size});
""")
