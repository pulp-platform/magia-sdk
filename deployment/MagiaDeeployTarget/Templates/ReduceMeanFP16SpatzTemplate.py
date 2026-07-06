from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation
import numpy as np

class _MagiaReduceMeanFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        axes = operatorRepresentation.get('axes', [1])
        axis = axes[0]
        if axis < 0:
            axis += len(data_in.shape)

        input_shape = data_in.shape

        outer_dim = int(np.prod(input_shape[:axis])) if axis > 0 else 1
        reduce_dim = int(input_shape[axis])
        inner_dim = int(np.prod(input_shape[axis + 1:])) if axis < len(input_shape) - 1 else 1

        operatorRepresentation['outer_dim'] = outer_dim
        operatorRepresentation['reduce_dim'] = reduce_dim
        operatorRepresentation['inner_dim'] = inner_dim

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaReduceMeanFP16Spatz("""
// Magia ReduceMean FP16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_reducemean_fp16_spatz(${data_in}, ${data_out}, ${outer_dim}, ${reduce_dim}, ${inner_dim});
""")
