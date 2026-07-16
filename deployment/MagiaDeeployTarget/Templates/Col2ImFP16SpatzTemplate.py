from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaCol2ImFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        ctxt.lookup(operatorRepresentation['data_in'])
        ctxt.lookup(operatorRepresentation['data_out'])

        col_dims = operatorRepresentation['col_dims']
        operatorRepresentation['l_len'] = col_dims[0] * col_dims[1]

        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaCol2ImFP16Spatz("""
// Magia Col2Im FP16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
#ifdef ENABLE_NODE_LOGS
printf("[CV32 (%d)] Running node: ${nodeName} (${nodeOp})\\n", get_hartid());
#endif
MAGIA_col2im_fp16_spatz(${data_in},  ${data_out},  ${batch_size},  ${channels},  ${image_shape[0]},  ${image_shape[1]},  ${block_shape[0]},  ${block_shape[1]},  ${pads[0]},  ${pads[1]},  ${strides[0]},  ${strides[1]},  ${dilations[0]},  ${dilations[1]},  ${l_len});
""")
