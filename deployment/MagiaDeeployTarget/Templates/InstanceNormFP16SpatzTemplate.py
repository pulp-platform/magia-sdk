from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaInstanceNormFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt.lookup(operatorRepresentation['data_in'])
        ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaInstanceNormFP16Spatz("""
// Magia InstanceNorm FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_instancenorm_fp16_spatz(${data_in}, ${data_out}, ${scale}, ${bias}, ${epsilon});
""")
