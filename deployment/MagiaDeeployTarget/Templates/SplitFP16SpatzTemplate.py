import numpy as np

from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer

class _MagiaSplitFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        if 'split' in operatorRepresentation:
            ctxt.globalObjects[operatorRepresentation['split']]._deploy = False
            ctxt.globalObjects[operatorRepresentation['split']]._live = False

        bufferIn = ctxt.lookup(operatorRepresentation['data_in'])
        assert isinstance(bufferIn, VariableBuffer)
        axis = operatorRepresentation['axis']
        if axis < 0:
            axis += len(bufferIn.shape)

        # Each output is a contiguous sub-region of the input only if the dimensions before the split
        # axis collapse to 1 (otherwise the slices interleave and aliasing is invalid).
        outer = int(np.prod(bufferIn.shape[:axis])) if axis > 0 else 1
        assert outer == 1, f"Split alias requires contiguous slices (dims before axis {axis} must be 1), got {bufferIn.shape}"

        offset = 0
        for idx in range(operatorRepresentation['num_outputs']):
            bufferOut = ctxt.lookup(operatorRepresentation[f'data_out_{idx}'])
            assert isinstance(bufferOut, VariableBuffer)
            shape = bufferOut.shape
            inner = int(np.prod(shape[axis + 1:])) if axis + 1 < len(shape) else 1

            bufferIn.aliases.add(bufferOut.name)
            bufferOut.aliases.add(bufferIn.name)
            bufferOut._alias = ctxt._mangle(bufferIn.name) + (f" + {offset}" if offset else "")

            offset += shape[axis] * inner

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaSplitFP16Spatz("""
// Magia Split FP16 Spatz (Name: ${nodeName}, Op: ${nodeOp}) -- no-op: each output aliases a slice of ${data_in}
#ifdef ENABLE_NODE_LOGS
printf("[CV32 (%d)] Running node: ${nodeName} (${nodeOp})\\n", get_hartid());
#endif
""")
