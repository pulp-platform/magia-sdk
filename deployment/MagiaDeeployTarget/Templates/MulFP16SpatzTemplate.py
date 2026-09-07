from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaMulFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt.lookup(operatorRepresentation['A'])
        ctxt.lookup(operatorRepresentation['B'])
        ctxt.lookup(operatorRepresentation['C'])

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaMulFP16Spatz("""
// Magia Mul FP16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
#ifdef ENABLE_NODE_LOGS
printf("[CV32 (%d)] Running node: ${nodeName} (${nodeOp})\\n", get_hartid());
#endif
MAGIA_mul_fp16_spatz(${A}, ${B}, ${C}, ${size});
""")
