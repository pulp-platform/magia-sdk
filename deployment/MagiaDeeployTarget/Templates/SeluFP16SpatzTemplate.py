
from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaSELUFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt.lookup(operatorRepresentation['data_in'])
        ctxt.lookup(operatorRepresentation['data_out'])
        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaSELUFP16Spatz("""
// Magia SELU FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_selu_fp16_spatz(${data_in}, ${data_out}, ${size}, ${alpha}, ${gamma});
""")
