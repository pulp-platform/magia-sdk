
from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaLeakyReluFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt.lookup(operatorRepresentation['data_in'])
        ctxt.lookup(operatorRepresentation['data_out'])
        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaLeakyReluFP16Spatz("""
// Magia LeakyRelu FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_leakyrelu_fp16_spatz(${data_in}, ${data_out}, ${size}, ${alpha});
""")
