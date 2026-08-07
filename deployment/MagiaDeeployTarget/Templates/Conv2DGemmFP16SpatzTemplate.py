# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaConv2DGemmFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['input_shape'] = "{" + ", ".join(map(str, data_in.shape)) + "}"
        operatorRepresentation['output_shape'] = "{" + ", ".join(map(str, data_out.shape)) + "}"
        operatorRepresentation['has_bias_c'] = int('bias' in operatorRepresentation)
        if 'bias' not in operatorRepresentation:
            operatorRepresentation['bias'] = 'NULL'
        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaConv2DGemmFP16Spatz("""
// Magia Conv2D (im2col + GEMM) FP16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
#ifdef ENABLE_NODE_LOGS
printf("[CV32 (%d)] Running node: ${nodeName} (${nodeOp})\\n", get_hartid());
#endif
MAGIA_conv2dgemm_fp16_spatz(${data_in}, ${weight}, ${bias}, ${data_out}, (uint32_t[])${input_shape}, (uint32_t[])${output_shape}, ${kernel_shape[0]}, ${kernel_shape[1]}, ${strides[0]}, ${strides[1]}, ${pads[0]}, ${pads[1]}, ${group}, ${has_bias_c});
""")
