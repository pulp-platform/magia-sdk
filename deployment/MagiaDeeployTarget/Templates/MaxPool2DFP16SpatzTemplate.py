from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaMaxPool2DFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['input_shape'] = "{" + ", ".join(map(str, data_in.shape)) + "}"
        operatorRepresentation['output_shape'] = "{" + ", ".join(map(str, data_out.shape)) + "}"

        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaMaxPool2DFP16Spatz("""
// Magia MaxPool2D FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_maxpool2d_fp16_spatz(${data_in}, ${data_out}, ${dim_kernel_x}, ${dim_kernel_y}, ${stride_x}, ${stride_y}, ${padding_x}, ${padding_y}, (uint32_t[])${input_shape}, (uint32_t[])${output_shape});
""")
