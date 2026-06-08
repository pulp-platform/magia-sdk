# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NetworkContext, NodeMapper, \
    NodeTemplate, StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.Targets.Generic.Layers import AddLayer, AveragePoolLayer, BatchNormalizationLayer, CeilLayer, ClipLayer, ConcatLayer, ConvLayer, ConvTransposeLayer, DivLayer, ExpLayer, FloorLayer, GatherLayer, GELULayer, GEMMLayer, GlobalAveragePoolLayer, GlobalMaxPoolLayer, GroupNormLayer, HardSigmoidLayer, HardSwishLayer, InstanceNormLayer, LayerNormLayer, MatMulLayer, MaxPoolLayer, ReluLayer, SigmoidLayer, SoftmaxLayer, SubLayer, SwishLayer, TransposeLayer
from Deeploy.Targets.Generic.Parsers import AddParser, AveragePool2DParser, BatchNormParser, CeilParser, ClipParser, ConcatParser, Conv2DParser, ConvTransposeParser, DivParser, ExpParser, FloorParser, GatherParser, GELUParser, GEMMParser, GlobalAveragePoolParser, GlobalMaxPoolParser, GroupNormParser, HardSigmoidParser, HardSwishParser, InstanceNormParser, LayerNormParser, MatMulParser, MaxPool2DParser, ReluParser, SigmoidParser, SoftmaxParser, SubParser, SwishParser, TransposeParser
from Deeploy.Targets.Generic.Templates import AllocateTemplate as BasicAllocateTemplate
from MagiaDeeployTarget.Bindings import MagiaAddBindings, MagiaAddFp16Bindings, MagiaAveragePool2DFp16Bindings, MagiaBatchNormFp16Bindings, MagiaCeilFp16Bindings, MagiaClipFp16Bindings, MagiaConcatFp16Bindings, MagiaConv2DFp16Bindings, MagiaConvTransposeBindings, MagiaDivFp16Bindings, MagiaExpFp16Bindings, MagiaFloorFp16Bindings, MagiaGatherFp16Bindings, MagiaGeluFp16Bindings, MagiaGemmFp16Bindings, MagiaGlobalAveragePoolFp16Bindings, MagiaGlobalMaxPoolFp16Bindings, MagiaGroupNormFp16Bindings, MagiaHardSigmoidFp16Bindings, MagiaHardSwishFp16Bindings, MagiaInstanceNormFp16Bindings, MagiaLayerNormFp16Bindings, MagiaMatMulFp16Bindings, MagiaMaxPool2DFp16Bindings, MagiaReluFp16Bindings, MagiaSigmoidFp16Bindings, MagiaSoftmaxFp16Bindings, MagiaSubFp16Bindings, MagiaSwishFp16Bindings, MagiaTransposeFp16Bindings
from MagiaDeeployTarget.Templates import AllocateTemplate, FreeTemplate


AddMapper = NodeMapper(AddParser(), MagiaAddBindings + MagiaAddFp16Bindings)
AveragePool2DPMapper = NodeMapper(AveragePool2DParser(), MagiaAveragePool2DFp16Bindings)
BatchNormMapper = NodeMapper(BatchNormParser(), MagiaBatchNormFp16Bindings)
CeilMapper = NodeMapper(CeilParser(), MagiaCeilFp16Bindings)
ClipMapper = NodeMapper(ClipParser(), MagiaClipFp16Bindings)
ConcatMapper = NodeMapper(ConcatParser(), MagiaConcatFp16Bindings)
Conv2DMapper = NodeMapper(Conv2DParser(), MagiaConv2DFp16Bindings)
ConvTransposeMapper = NodeMapper(ConvTransposeParser(), MagiaConvTransposeBindings)
DivMapper = NodeMapper(DivParser(), MagiaDivFp16Bindings)
ExpMapper = NodeMapper(ExpParser(), MagiaExpFp16Bindings)
FloorMapper = NodeMapper(FloorParser(), MagiaFloorFp16Bindings)
GatherMapper = NodeMapper(GatherParser(), MagiaGatherFp16Bindings)
GeluMapper = NodeMapper(GELUParser(), MagiaGeluFp16Bindings)
GemmMapper = NodeMapper(GEMMParser(), MagiaGemmFp16Bindings)
GlobalAveragePoolMapper = NodeMapper(GlobalAveragePoolParser(), MagiaGlobalAveragePoolFp16Bindings)
GlobalMaxPoolMapper = NodeMapper(GlobalMaxPoolParser(), MagiaGlobalMaxPoolFp16Bindings)
GroupNormMapper = NodeMapper(GroupNormParser(), MagiaGroupNormFp16Bindings)
HardSigmoidMapper = NodeMapper(HardSigmoidParser(), MagiaHardSigmoidFp16Bindings)
HardSwishMapper = NodeMapper(HardSwishParser(), MagiaHardSwishFp16Bindings)
InstanceNormMapper = NodeMapper(InstanceNormParser(), MagiaInstanceNormFp16Bindings)
LayerNormMapper = NodeMapper(LayerNormParser(), MagiaLayerNormFp16Bindings)
MatMulMapper = NodeMapper(MatMulParser(), MagiaMatMulFp16Bindings)
MaxPool2DMapper = NodeMapper(MaxPool2DParser(), MagiaMaxPool2DFp16Bindings)
ReluMapper = NodeMapper(ReluParser(), MagiaReluFp16Bindings)
SigmoidMapper = NodeMapper(SigmoidParser(), MagiaSigmoidFp16Bindings)
SoftmaxMapper = NodeMapper(SoftmaxParser(), MagiaSoftmaxFp16Bindings)
SubMapper = NodeMapper(SubParser(), MagiaSubFp16Bindings)
SwishMapper = NodeMapper(SwishParser(), MagiaSwishFp16Bindings)
TransposeMapper = NodeMapper(TransposeParser(), MagiaTransposeFp16Bindings)

MagiaMapping = {
    'Add': AddLayer([AddMapper]),
    'AveragePool': AveragePoolLayer([AveragePool2DPMapper]),
    'BatchNormalization': BatchNormalizationLayer([BatchNormMapper]),
    'Ceil': CeilLayer([CeilMapper]),
    'Clip': ClipLayer([ClipMapper]),
    'Concat': ConcatLayer([ConcatMapper]),
    'Conv': ConvLayer([Conv2DMapper]),
    'ConvTranspose': ConvTransposeLayer([ConvTransposeMapper]),
    'Div': DivLayer([DivMapper]),
    'Exp': ExpLayer([ExpMapper]),
    'Floor': FloorLayer([FloorMapper]),
    'Gather': GatherLayer([GatherMapper]),
    'Gelu': GELULayer([GeluMapper]),
    'Gemm': GEMMLayer([GemmMapper]),
    'GlobalAveragePool': GlobalAveragePoolLayer([GlobalAveragePoolMapper]),
    'GlobalMaxPool': GlobalMaxPoolLayer([GlobalMaxPoolMapper]),
    'GroupNormalization': GroupNormLayer([GroupNormMapper]),
    'HardSigmoid': HardSigmoidLayer([HardSigmoidMapper]),
    'HardSwish': HardSwishLayer([HardSwishMapper]),
    'InstanceNormalization': InstanceNormLayer([InstanceNormMapper]),
    'LayerNormalization': LayerNormLayer([LayerNormMapper]),
    'MatMul': MatMulLayer([MatMulMapper]),
    'MaxPool': MaxPoolLayer([MaxPool2DMapper]),
    'Relu': ReluLayer([ReluMapper]),
    'Sigmoid': SigmoidLayer([SigmoidMapper]),
    'Softmax': SoftmaxLayer([SoftmaxMapper]),
    'Sub': SubLayer([SubMapper]),
    'Swish': SwishLayer([SwishMapper]),
    'Transpose': TransposeLayer([TransposeMapper]),
}

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
