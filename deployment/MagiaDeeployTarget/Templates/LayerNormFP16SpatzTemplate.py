from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaLayerNormFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        ctxt.lookup(operatorRepresentation['weight'])
        ctxt.lookup(operatorRepresentation['bias'])
        ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['input_shape'] = "{" + ", ".join(map(str, data_in.shape)) + "}"

        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaLayerNormFP16Spatz("""
// Magia LayerNorm FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_layernorm_fp16_spatz(${data_in}, ${weight}, ${bias}, ${epsilon}, (uint32_t[])${input_shape}, ${data_out});
""")
