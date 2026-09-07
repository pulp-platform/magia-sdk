from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaLayerNormFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        ctxt.lookup(operatorRepresentation['weight'])
        ctxt.lookup(operatorRepresentation['bias'])
        ctxt.lookup(operatorRepresentation['data_out'])

        # Deeploy's LayerNorm parser ignores the ONNX attribute and always assumes axis == -1 (see lastDimLength)
        data_in_shape = data_in.shape
        operatorRepresentation['input_shape'] = "{" + ", ".join(map(str, data_in_shape)) + "}"
        operatorRepresentation['rank'] = len(data_in_shape)

        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaLayerNormFP16Spatz("""
// Magia LayerNorm FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
#ifdef ENABLE_NODE_LOGS
printf("[CV32 (%d)] Running node: ${nodeName} (${nodeOp})\\n", get_hartid());
#endif
MAGIA_layernorm_fp16_spatz(${data_in}, ${weight}, ${bias}, ${epsilon}, (uint32_t[])${input_shape}, ${rank}, ${data_out});
""")
