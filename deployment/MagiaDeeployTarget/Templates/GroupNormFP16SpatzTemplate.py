from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaGroupNormFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        data_in = ctxt.lookup(operatorRepresentation['data_in'])

        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaGroupNormFP16Spatz("""
// Magia GroupNorm FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_groupnorm_fp16_spatz(${data_in}, ${data_out}, ${scale}, ${bias}, ${batch_size}, ${num_channels}, ${spatial}, ${num_groups}, ${epsilon});
""")
