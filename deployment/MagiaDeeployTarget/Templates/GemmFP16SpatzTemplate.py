from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaGemmFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        A = ctxt.lookup(operatorRepresentation['A'])
        B = ctxt.lookup(operatorRepresentation['B'])
        C = ctxt.lookup(operatorRepresentation['C'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['A_shape'] = "{" + ", ".join(map(str, A.shape)) + "}"
        operatorRepresentation['B_shape'] = "{" + ", ".join(map(str, B.shape)) + "}"
        operatorRepresentation['C_shape'] = "{" + ", ".join(map(str, C.shape)) + "}"
        operatorRepresentation['Y_shape'] = "{" + ", ".join(map(str, data_out.shape)) + "}"

        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaGemmFP16Spatz("""
// Magia Gemm FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_gemm_fp16_spatz(${A}, ${B}, ${C}, ${alpha}, ${beta}, ${transA}, ${transB}, (uint32_t[])${A_shape}, (uint32_t[])${B_shape}, (uint32_t[])${C_shape}, (uint32_t[])${Y_shape}, ${data_out});
""")
