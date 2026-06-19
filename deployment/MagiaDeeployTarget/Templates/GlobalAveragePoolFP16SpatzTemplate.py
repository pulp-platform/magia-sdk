from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaGlobalAveragePoolFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['input_shape'] = "{" + ", ".join(map(str, data_in.shape)) + "}"
        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaGlobalAveragePoolFP16Spatz("""
// Magia GlobalAveragePool FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_globalaveragepool_fp16_spatz(${data_in}, ${data_out}, (uint32_t[])${input_shape});
""")
