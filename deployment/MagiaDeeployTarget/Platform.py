# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NetworkContext, NodeMapper, \
    NodeTemplate, StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.Targets.Generic.Layers import AddLayer
from Deeploy.Targets.Generic.Parsers import AddParser
from Deeploy.Targets.Generic.Templates import AllocateTemplate as BasicAllocateTemplate
from MagiaDeeployTarget.Bindings import MagiaAddBindings
from MagiaDeeployTarget.Templates import AllocateTemplate, FreeTemplate


AddMapper = NodeMapper(AddParser(), MagiaAddBindings)

MagiaMapping = {'Add': AddLayer([AddMapper])}


class MagiaVariableBuffer(VariableBuffer):

    initTemplate = AllocateTemplate.magiaInitTemplate
    allocTemplate = AllocateTemplate.magiaAllocateTemplate
    deallocTemplate = FreeTemplate.magiaFreeTemplate

    def _bufferRepresentation(self):
        buffRepr = {
            "type": self._instance,
            "name": self.name,
            "size": int(np.prod(self.shape)),
            "_memoryLevel": getattr(self, "_memoryLevel", None),
        }
        return buffRepr


class MagiaTransientBuffer(TransientBuffer):

    initTemplate = AllocateTemplate.magiaInitTemplate
    allocTemplate = AllocateTemplate.magiaAllocateTemplate
    deallocTemplate = FreeTemplate.magiaFreeTemplate

    def _bufferRepresentation(self):
        buffRepr = {
            "type": self._type,
            "name": self.name,
            "size": self.size,
            "_memoryLevel": getattr(self, "_memoryLevel", None),
        }
        return buffRepr


class MagiaConstantBuffer(ConstantBuffer):

    initTemplate = AllocateTemplate.magiaGlobalInitTemplate
    allocTemplate = AllocateTemplate.magiaGlobalAllocateTemplate
    deallocTemplate = FreeTemplate.magiaGlobalTemplate

    def _bufferRepresentation(self):
        buffRepr = super()._bufferRepresentation()
        buffRepr["_memoryLevel"] = getattr(self, "_memoryLevel", None)
        return buffRepr


class MagiaStructBuffer(StructBuffer):

    initTemplate = BasicAllocateTemplate.referenceStructInitTemplate
    allocTemplate = BasicAllocateTemplate.referenceStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


MagiaOptimizer = TopologyOptimizer(
    [
        # Insert here the ONNX optimization passes.
    ],
    name = "MagiaOptimizer")

_includeList = ["tile.h", "idma.h", "redmule.h", "eventunit.h"]


class MagiaMeshEngine(DeploymentEngine):

    def __init__(self,
                 name: str,
                 Mapping = MagiaMapping,
                 initCode: str = "",
                 includeList: list[str] = _includeList,
                 n_tiles: int = 4) -> None:
        super().__init__(name, Mapping, initCode, includeList)
        self.n_tiles = n_tiles


class MagiaPlatform(DeploymentPlatform):

    def __init__(self,
                 engines = [MagiaMeshEngine("MagiaMesh")],
                 variableBuffer = MagiaVariableBuffer,
                 constantBuffer = MagiaConstantBuffer,
                 structBuffer = MagiaStructBuffer,
                 transientBuffer = MagiaTransientBuffer) -> None:
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)
