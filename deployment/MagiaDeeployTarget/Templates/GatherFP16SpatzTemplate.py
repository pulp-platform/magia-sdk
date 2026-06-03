# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple
import numpy as np
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaGatherFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        shape = data_in.shape
        axis = operatorRepresentation['axis']

        gather_dim_size = int(shape[axis])
        operatorRepresentation['gather_dim_size'] = gather_dim_size

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaGatherFP16Spatz("""
// Magia Gather FP16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_gather_fp16_spatz(${data_in}, ${data_out}, ${batch}, ${gather_dim_size}, ${axis_length}, ${index});
""")
