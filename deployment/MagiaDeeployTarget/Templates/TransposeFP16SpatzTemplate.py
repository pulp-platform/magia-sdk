# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

class _MagiaTransposeFP16Spatz(NodeTemplate):
    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_in_shape = data_in.shape
        data_out_shape = operatorRepresentation['data_out_shape']
        perm = operatorRepresentation['perm']

        assert perm[0] == 0, f"Transpose Spatz kernel requires perm[0] == 0 (axis-0 sharding). Got perm={list(perm)}."

        rank = len(data_in_shape)
        iterations = data_out_shape[0]
        operatorRepresentation['rank'] = rank
        operatorRepresentation['iterations'] = iterations

        # from python list to C array
        operatorRepresentation['input_shape'] = "{" + ", ".join(map(str, data_in_shape)) + "}"
        operatorRepresentation['output_shape'] = "{" + ", ".join(map(str, data_out_shape)) + "}"
        operatorRepresentation['c_perm'] = "{" + ", ".join(map(str, perm)) + "}"

        return ctxt, operatorRepresentation, []

referenceTemplate = _MagiaTransposeFP16Spatz("""
// Magia Transpose FP16 Spatz (Name: ${nodeName}, Op: ${nodeOp})
MAGIA_transpose_fp16_spatz(${data_in}, ${data_out}, (uint32_t[]) ${c_perm}, (uint32_t[])${input_shape}, (uint32_t[])${output_shape}, ${rank}, ${iterations});
""")
