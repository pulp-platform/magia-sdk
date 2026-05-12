
from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaDivFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt.lookup(operatorRepresentation['input1'])
        ctxt.lookup(operatorRepresentation['input2'])
        ctxt.lookup(operatorRepresentation['output'])
        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaDivFP16Spatz("""
// Magia Div FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_div_fp16_spatz(${input1}, ${input2}, ${output}, ${size});
""")
