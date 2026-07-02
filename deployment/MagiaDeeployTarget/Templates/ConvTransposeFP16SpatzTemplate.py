# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaConvTransposeFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt.lookup(operatorRepresentation['weight'])
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])
        operatorRepresentation['offset'] = 0

        operatorRepresentation['_input_shape'] = "{" + ", ".join(map(str, data_in.shape)) + "}"
        N = operatorRepresentation['batch_size']
        C = operatorRepresentation['feature_maps']
        H = operatorRepresentation['output_shape'][0]
        W = operatorRepresentation['output_shape'][1]

        operatorRepresentation['_output_shape'] = f"{{ {N}, {C}, {H}, {W} }}"

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaConvTransposeFP16Spatz("""
// Magia ConvTranspose FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_convtranspose_fp16_spatz(${data_in}, ${weight}, ${data_out}, (uint32_t[])${_input_shape}, (uint32_t[])${_output_shape}, ${kernel_shape[0]}, ${kernel_shape[1]}, ${strides[0]}, ${strides[1]}, ${pads[0]}, ${pads[1]}, ${group});
""")
