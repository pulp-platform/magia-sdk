from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaGlobalMaxPoolFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt.lookup(operatorRepresentation['data_in'])

        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaGlobalMaxPoolFP16Spatz("""
// Magia GlobalMaxPool FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_globalmaxpool_fp16_spatz(${data_in}, ${data_out});
""")
