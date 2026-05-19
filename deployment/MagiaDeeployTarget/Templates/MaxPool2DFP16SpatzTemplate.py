from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaMaxPool2DFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt.lookup(operatorRepresentation['data_in'])
        ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaMaxPool2DFP16Spatz("""
// Magia MaxPool2D FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_maxpool2d_fp16_spatz(${data_in}, ${data_out}, ${dim_kernel_x}, ${dim_kernel_y}, ${stride_x}, ${stride_y}, ${padding_x}, ${padding_y});
""")
