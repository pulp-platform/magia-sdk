from typing import Dict, List, Tuple
import numpy as np
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaScatterFP16Spatz(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation):

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        indices = ctxt.lookup(operatorRepresentation['indices'])
        updates = ctxt.lookup(operatorRepresentation['updates'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        data_shape = data_in.shape
        indices_shape = indices.shape
        axis = operatorRepresentation['axis']

        if axis < 0:
            axis += len(data_shape)

        outer_size = int(np.prod(data_shape[:axis])) if axis > 0 else 1
        inner_size = int(np.prod(data_shape[axis + 1:])) if axis < len(data_shape) - 1 else 1

        data_axis_dim = int(data_shape[axis])
        indices_axis_dim = int(indices_shape[axis])

        operatorRepresentation['outer_size'] = outer_size
        operatorRepresentation['inner_size'] = inner_size
        operatorRepresentation['data_axis_dim'] = data_axis_dim
        operatorRepresentation['indices_axis_dim'] = indices_axis_dim

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaScatterFP16Spatz("""
// Magia Scatter FP16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
#ifdef ENABLE_NODE_LOGS
printf("[CV32 (%d)] Running node: ${nodeName} (${nodeOp})\\n", get_hartid());
#endif
MAGIA_scatter_fp16_spatz(${data_in}, ${indices}, ${updates}, ${data_out}, ${outer_size}, ${inner_size}, ${axis}, ${data_axis_dim}, ${indices_axis_dim});
""")
