# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from Deeploy.DeeployTypes import ONNXLayer, OperatorRepresentation, Shape
from Deeploy.Targets.Generic.Layers import SingleOperationPerElementLayer


class ElementwiseBroadcastLayer(SingleOperationPerElementLayer):

    def computeShapes(
            self,
            inputShapes: Shape,
            outputShapes: Shape,
            operatorRepresentation: OperatorRepresentation,
            channels_first: bool
        ) -> tuple[list[Shape], list[Shape]]:

        # TODO: This is not optimal and should be improved in the future.
        # This is a naive solution that broadcasts input shapes by
        # hoisting buffers as big as the broadcast shape (even when on of the
        # input is a scalar).

        # turn scalar inputs into 1-dimensional tensors of shape (1,)
        input_shapes = [tuple(shape) if len(shape) > 0 else (1,)
                         for shape in inputShapes[:2]]

        # compute broadcast shape of the two inputs
        # TODO: move broadcastability check to the Parser
        try:
            broadcast_shape = list(np.broadcast_shapes(*input_shapes))
        except ValueError as e:
            _str = f"Node {self.node.name}: operand shapes {input_shapes} " \
                    "are not broadcastable!"
            raise ValueError(_str) from e

        # apply broadcast shape to all inputs and the output
        for idx in range(len(input_shapes)):
            inputShapes[idx] = list(broadcast_shape)
        outputShapes = [list(broadcast_shape)]

        return (inputShapes, outputShapes)

AddLayer = ElementwiseBroadcastLayer
SubLayer = ElementwiseBroadcastLayer
MulLayer = ElementwiseBroadcastLayer
DivLayer = ElementwiseBroadcastLayer