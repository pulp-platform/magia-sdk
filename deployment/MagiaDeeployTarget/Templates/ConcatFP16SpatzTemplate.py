# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple
import numpy as np
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaConcatFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        axis = operatorRepresentation['axis']

        assert axis != 0, f"axis == 0 not yet supported"

        inputs = []
        idx = 1
        while f'data_in_{idx}' in operatorRepresentation:
            inputs.append(operatorRepresentation[f'data_in_{idx}'])
            idx += 1

        transfer_lens = []
        iterations = None
        for name in inputs:
            shape = ctxt.lookup(name).shape
            transfer_lens.append(int(np.prod(shape[axis:])))
            it = int(np.prod(shape[:axis]))
            assert iterations is None or it == iterations, f"iterations {it} is not {iterations}; concat can't be applied!"
            iterations = it

        operatorRepresentation['num_inputs'] = len(inputs)
        operatorRepresentation['iterations'] = iterations
        operatorRepresentation['inputs'] = "{" + ", ".join(ctxt._mangle(name) for name in inputs) + "}"
        operatorRepresentation['lens'] = "{" + ", ".join(map(str, transfer_lens)) + "}"

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaConcatFP16Spatz("""
// Magia Concat FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
#ifdef ENABLE_NODE_LOGS
printf("[CV32 (%d)] Running node: ${nodeName} (${nodeOp})\\n", get_hartid());
#endif
MAGIA_concat_fp16_spatz((const float16 *[])${inputs}, (uint32_t[])${lens}, ${num_inputs}, ${data_out}, ${iterations});
""")
