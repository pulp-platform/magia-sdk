from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation, VariableBuffer

class _MagiaReshapeFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        if 'shape' in operatorRepresentation:
            ctxt.globalObjects[operatorRepresentation['shape']]._deploy = False
            ctxt.globalObjects[operatorRepresentation['shape']]._live = False

        bufferIn = ctxt.lookup(operatorRepresentation['data_in'])
        bufferOut = ctxt.lookup(operatorRepresentation['data_out'])
        assert isinstance(bufferIn, VariableBuffer) and isinstance(bufferOut, VariableBuffer)

        bufferIn.aliases.add(bufferOut.name)
        bufferOut.aliases.add(bufferIn.name)
        bufferOut._alias = ctxt._mangle(bufferIn.name)

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaReshapeFP16Spatz("""
// Magia Reshape FP16 Spatz (Name: ${nodeName}, Op: ${nodeOp}) -- no-op: ${data_out} aliases ${data_in}
#ifdef ENABLE_NODE_LOGS
printf("[CV32 (%d)] Running node: ${nodeName} (${nodeOp})\\n", get_hartid());
#endif
""")
