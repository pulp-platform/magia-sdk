from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation
import numpy as np

class _MagiaSliceFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        axes_tensor = ctxt.lookup(operatorRepresentation['axes'])
        starts_tensor = ctxt.lookup(operatorRepresentation['starts'])

        axis = int(axes_tensor.values[0])
        start_idx = int(starts_tensor.values[0])

        input_shape = data_in.shape
        output_shape = data_out.shape

        outer_dim = int(np.prod(input_shape[:axis])) if axis > 0 else 1
        slice_dim = int(input_shape[axis])
        out_slice_dim = int(output_shape[axis])
        inner_dim = int(np.prod(input_shape[axis + 1:])) if axis < len(input_shape) - 1 else 1

        operatorRepresentation['outer_dim'] = outer_dim
        operatorRepresentation['slice_dim'] = slice_dim
        operatorRepresentation['out_slice_dim'] = out_slice_dim
        operatorRepresentation['inner_dim'] = inner_dim
        operatorRepresentation['start_idx'] = start_idx

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaSliceFP16Spatz("""
// Magia Slice FP16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
#ifdef ENABLE_NODE_LOGS
printf("[CV32 (%d)] Running node: ${nodeName} (${nodeOp})\\n", get_hartid());
#endif
MAGIA_slice_fp16_spatz(${data_in}, ${data_out}, ${outer_dim}, ${slice_dim}, ${inner_dim}, ${start_idx}, ${out_slice_dim});
""")
