# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import onnx_graphsurgeon as gs
from Deeploy.Targets.Generic.Parsers import Conv2DParser


class DWConv2DParser(Conv2DParser):
    """Accepts only depthwise convolutions, the rest falls through to the next mapper."""

    def parseNode(self, node: gs.Node) -> (bool):

        if not super().parseNode(node):
            return False

        # Depthwise: a single input channel per group in the weight [C_out, C_in / group, K_h, K_w]
        return self.operatorRepresentation['group'] > 1 and node.inputs[1].shape[1] == 1
