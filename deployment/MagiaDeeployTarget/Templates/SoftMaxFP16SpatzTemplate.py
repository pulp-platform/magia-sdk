from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaSoftmaxFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        ctxt.lookup(operatorRepresentation['data_out'])

        shape = list(data_in.shape)
        row_len = shape[-1]
        outer = 1
        for dim in shape[:-1]:
            outer *= dim
        shape4 = [1, 1, outer, row_len]

        operatorRepresentation['input_shape'] = "{" + ", ".join(map(str, shape4)) + "}"
        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaSoftmaxFP16Spatz("""
// Magia Softmax FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
#ifdef ENABLE_NODE_LOGS
printf("[CV32 (%d)] Running node: ${nodeName} (${nodeOp})\\n", get_hartid());
#endif
MAGIA_softmax_fp16_spatz(${data_in}, ${data_out}, (uint32_t[])${input_shape});
""")
