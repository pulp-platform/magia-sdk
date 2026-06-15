from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaAveragePool2DFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['input_shape'] = "{" + ", ".join(map(str, data_in.shape)) + "}"
        operatorRepresentation['output_shape'] = "{" + ", ".join(map(str, data_out.shape)) + "}"

        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaAveragePool2DFP16Spatz("""
// Magia AveragePool2D FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_averagepool2d_fp16_spatz(${data_in}, ${data_out}, (uint32_t[])${input_shape}, (uint32_t[])${output_shape}, ${kernel_shape[0]}, ${kernel_shape[1]}, ${strides[0]}, ${strides[1]}, ${pads[0]}, ${pads[1]});
""")
