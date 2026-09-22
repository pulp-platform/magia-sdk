from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaResizeFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['batch_size'] = data_in.shape[0]
        operatorRepresentation['channels']   = data_in.shape[1]
        operatorRepresentation['in_h']  = data_in.shape[2]
        operatorRepresentation['in_w']  = data_in.shape[3]
        operatorRepresentation['out_h'] = data_out.shape[2]
        operatorRepresentation['out_w'] = data_out.shape[3]

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaResizeFP16Spatz("""
// Magia Resize Nearest Neighbor FP16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
#ifdef ENABLE_NODE_LOGS
printf("[CV32 (%d)] Running node: ${nodeName} (${nodeOp})\\n", get_hartid());
#endif
MAGIA_resize_fp16_spatz(${data_in}, ${data_out}, ${batch_size}, ${channels}, ${in_h}, ${in_w}, ${out_h}, ${out_w});
""")
