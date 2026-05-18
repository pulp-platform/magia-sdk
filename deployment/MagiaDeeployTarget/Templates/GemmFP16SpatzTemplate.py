
from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaGemmFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt.lookup(operatorRepresentation['A'])
        ctxt.lookup(operatorRepresentation['B'])
        ctxt.lookup(operatorRepresentation['C'])
        ctxt.lookup(operatorRepresentation['data_out'])
        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaGemmFP16Spatz("""
// Magia Gemm FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_gemm_fp16_spatz(${A}, ${B}, ${C}, ${alpha}, ${beta}, ${transA}, ${transB}, ${data_out});
""")
