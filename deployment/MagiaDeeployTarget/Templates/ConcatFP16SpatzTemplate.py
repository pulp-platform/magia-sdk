# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple
import numpy as np
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaConcatFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        dataIn1 = ctxt.lookup(operatorRepresentation['data_in_1'])
        dataIn2 = ctxt.lookup(operatorRepresentation['data_in_2'])

        assert "data_in_3" not in operatorRepresentation.keys(), "Concat with more than two inputs not implemented!"

        dataIn1Shape = dataIn1.shape
        dataIn2Shape = dataIn2.shape

        axis = operatorRepresentation['axis']

        assert axis != 0, f"axis == 0 not yet supported"

        in0_transfer_len = np.prod(dataIn1Shape[axis:])
        in1_transfer_len = np.prod(dataIn2Shape[axis:])

        iterations1 = np.prod(dataIn1Shape[:axis])
        iterations2 = np.prod(dataIn2Shape[:axis])

        assert iterations1 == iterations2, f"iterations1 {iterations1} is not iterations2 {iterations2}; concat can't be applied!"

        operatorRepresentation['iterations'] = iterations1
        operatorRepresentation['in0_transfer_len'] = in0_transfer_len
        operatorRepresentation['in1_transfer_len'] = in1_transfer_len

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaConcatFP16Spatz("""
// Magia Concat FP 16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_concat_fp16_spatz(${data_in_1}, ${data_in_2}, ${data_out}, ${in0_transfer_len}, ${in1_transfer_len}, ${axis}, ${iterations});
""")
