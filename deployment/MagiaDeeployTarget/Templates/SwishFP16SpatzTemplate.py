
from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaSwishFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt.lookup(operatorRepresentation['data_in'])
        ctxt.lookup(operatorRepresentation['data_out'])
        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaSwishFP16Spatz("""
// Magia Swish FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
#ifdef ENABLE_NODE_LOGS
printf("[CV32 (%d)] Running node: ${nodeName} (${nodeOp})\\n", get_hartid());
#endif
MAGIA_swish_fp16_spatz(${data_in}, ${data_out}, ${alpha}, ${size});
""")
