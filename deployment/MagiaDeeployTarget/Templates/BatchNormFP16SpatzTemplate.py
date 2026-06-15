from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaBatchNormFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        ctxt.lookup(operatorRepresentation['scale'])
        ctxt.lookup(operatorRepresentation['bias'])
        ctxt.lookup(operatorRepresentation['mean'])
        ctxt.lookup(operatorRepresentation['variance'])
        ctxt.lookup(operatorRepresentation['data_out'])

        operatorRepresentation['shape'] = "{" + ", ".join(map(str, data_in.shape)) + "}"

        operatorRepresentation['offset'] = 0

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaBatchNormFP16Spatz("""
// Magia BatchNorm FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_batchnorm_fp16_spatz(${data_in}, ${scale}, ${bias}, ${mean}, ${variance}, ${epsilon}, ${data_out}, (uint32_t[])${shape});
""")
