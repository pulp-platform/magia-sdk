from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation
import numpy as np

class _MagiaReshapeFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        ctxt.lookup(operatorRepresentation['shape'])
        ctxt.lookup(operatorRepresentation['data_out'])

        total_elements = int(np.prod(data_in.shape))
        operatorRepresentation['total_elements'] = total_elements

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaReshapeFP16Spatz("""
// Magia Reshape FP16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
#ifdef ENABLE_NODE_LOGS
printf("[CV32 (%d)] Running node: ${nodeName} (${nodeOp})\\n", get_hartid());
#endif
MAGIA_reshape_fp16_spatz(${data_in}, ${data_out}, ${total_elements});
""")
