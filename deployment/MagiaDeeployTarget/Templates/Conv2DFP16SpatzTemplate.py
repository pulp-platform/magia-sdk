# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaConv2DFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt.lookup(operatorRepresentation['data_in'])
        ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaConv2DFP16Spatz("""
// Magia Conv2D FP16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_conv2d_fp16_spatz(${data_in}, ${weight}, ${data_out}, ${kernel_shape[0]}, ${kernel_shape[1]}, ${strides[0]}, ${strides[1]}, ${pads[0]}, ${pads[1]}, ${group});
""")
