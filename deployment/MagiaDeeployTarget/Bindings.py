# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0


from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import float16_t, int8_t, int32_t, int64_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.Targets.Generic.TypeCheckers import AddChecker, BatchNormChecker, ConcatChecker, ConvChecker, DivChecker, DummyChecker, GatherChecker, GELUChecker, GEMMChecker, LayerNormChecker, MatMulChecker, MaxPoolChecker, MulChecker, PassThroughTypeChecker, ReluChecker, ReshapeChecker, SliceChecker, TransposeChecker
from MagiaDeeployTarget.Templates import AddFP16SpatzTemplate
from MagiaDeeployTarget.Templates import AddTemplate
from MagiaDeeployTarget.Templates import AveragePool2DFP16SpatzTemplate
from MagiaDeeployTarget.Templates import BatchNormFP16SpatzTemplate
from MagiaDeeployTarget.Templates import CeilFP16SpatzTemplate
from MagiaDeeployTarget.Templates import ClipFP16SpatzTemplate
from MagiaDeeployTarget.Templates import Col2ImFP16SpatzTemplate
from MagiaDeeployTarget.Templates import ConcatFP16SpatzTemplate
from MagiaDeeployTarget.Templates import Conv2DFP16SpatzTemplate
from MagiaDeeployTarget.Templates import ConvTransposeFP16SpatzTemplate
from MagiaDeeployTarget.Templates import DivFP16SpatzTemplate
from MagiaDeeployTarget.Templates import EluFP16SpatzTemplate
from MagiaDeeployTarget.Templates import ExpFP16SpatzTemplate
from MagiaDeeployTarget.Templates import FloorFP16SpatzTemplate
from MagiaDeeployTarget.Templates import GatherFP16SpatzTemplate
from MagiaDeeployTarget.Templates import GeluFP16SpatzTemplate
from MagiaDeeployTarget.Templates import GemmFP16SpatzTemplate
from MagiaDeeployTarget.Templates import GlobalAveragePoolFP16SpatzTemplate
from MagiaDeeployTarget.Templates import GlobalMaxPoolFP16SpatzTemplate
from MagiaDeeployTarget.Templates import GroupNormFP16SpatzTemplate
from MagiaDeeployTarget.Templates import HardSigmoidFP16SpatzTemplate
from MagiaDeeployTarget.Templates import HardSwishFP16SpatzTemplate
from MagiaDeeployTarget.Templates import InstanceNormFP16SpatzTemplate
from MagiaDeeployTarget.Templates import LayerNormFP16SpatzTemplate
from MagiaDeeployTarget.Templates import LeakyReluFP16SpatzTemplate
from MagiaDeeployTarget.Templates import MaxPool2DFP16SpatzTemplate
from MagiaDeeployTarget.Templates import MatMulFP16SpatzTemplate
from MagiaDeeployTarget.Templates import MulFP16SpatzTemplate
from MagiaDeeployTarget.Templates import ReluFP16SpatzTemplate
from MagiaDeeployTarget.Templates import ReshapeFP16SpatzTemplate
from MagiaDeeployTarget.Templates import ResizeFP16SpatzTemplate
from MagiaDeeployTarget.Templates import ScatterFP16SpatzTemplate
from MagiaDeeployTarget.Templates import SeluFP16SpatzTemplate
from MagiaDeeployTarget.Templates import SigmoidFP16SpatzTemplate
from MagiaDeeployTarget.Templates import SliceFP16SpatzTemplate
from MagiaDeeployTarget.Templates import SoftMaxFP16SpatzTemplate
from MagiaDeeployTarget.Templates import SubFP16SpatzTemplate
from MagiaDeeployTarget.Templates import SwishFP16SpatzTemplate
from MagiaDeeployTarget.Templates import TransposeFP16SpatzTemplate

BasicTransformer = CodeTransformation([])

MagiaAddBindings = [
    NodeBinding(
        AddChecker(
            [PointerClass(int8_t), PointerClass(int8_t)],
            [PointerClass(int32_t)]
        ),
        AddTemplate.referenceTemplate,
        BasicTransformer,
    ),
]

MagiaAddFp16Bindings = [
    NodeBinding(
        AddChecker(
            [PointerClass(float16_t), PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        AddFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    ),
]

MagiaAveragePool2DFp16Bindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        AveragePool2DFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    ),
]

MagiaBatchNormFp16Bindings = [
    NodeBinding(
        BatchNormChecker(
            [PointerClass(float16_t), PointerClass(float16_t), PointerClass(float16_t), PointerClass(float16_t), PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        BatchNormFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    ),
]

MagiaCeilFp16Bindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)]),
            CeilFP16SpatzTemplate.referenceTemplate,
            BasicTransformer,
        )
]

MagiaClipFp16Bindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float16_t), PointerClass(float16_t), PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        ClipFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    )
]

MagiaCol2ImFp16Bindings = [
    NodeBinding(
        PassThroughTypeChecker(
            [PointerClass(float16_t), PointerClass(int32_t), PointerClass(int32_t)],
            [PointerClass(float16_t)]
        ),
        Col2ImFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    )
]


MagiaConcatFp16Bindings = [
    NodeBinding(
        ConcatChecker(
            [PointerClass(float16_t), PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        ConcatFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    )
]

MagiaConv2DFp16Bindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(float16_t), PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        Conv2DFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    )
]

MagiaConvTransposeBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(float16_t), PointerClass(float16_t), PointerClass(float16_t)],
            [PointerClass(float16_t)],
        ),
        ConvTransposeFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    )
]

MagiaDivFp16Bindings = [
    NodeBinding(
        DivChecker(
            [PointerClass(float16_t), PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        DivFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    ),
]

MagiaEluFp16Bindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        EluFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    ),
]

MagiaExpFp16Bindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        ExpFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    ),
]

MagiaFloorFp16Bindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)]),
            FloorFP16SpatzTemplate.referenceTemplate,
            BasicTransformer,
        )
]

MagiaGatherFp16Bindings = [
    NodeBinding(
        GatherChecker(
            [PointerClass(float16_t), PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        GatherFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    )
]

MagiaGeluFp16Bindings = [
    NodeBinding(
        GELUChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        GeluFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    ),
]

MagiaGemmFp16Bindings = [
    NodeBinding(
        GEMMChecker(
            [PointerClass(float16_t), PointerClass(float16_t), PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        GemmFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    )
]

MagiaGlobalAveragePoolFp16Bindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        GlobalAveragePoolFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    )
]

MagiaGlobalMaxPoolFp16Bindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        GlobalMaxPoolFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    )
]

MagiaGroupNormFp16Bindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float16_t), PointerClass(float16_t), PointerClass(float16_t)],
            [PointerClass(float16_t)]),
            GroupNormFP16SpatzTemplate.referenceTemplate,
            BasicTransformer,
    )
]

MagiaHardSigmoidFp16Bindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        HardSigmoidFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    ),
]

MagiaHardSwishFp16Bindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        HardSwishFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    ),
]

MagiaInstanceNormFp16Bindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float16_t), PointerClass(float16_t), PointerClass(float16_t)],
            [PointerClass(float16_t)]),
            InstanceNormFP16SpatzTemplate.referenceTemplate,
            BasicTransformer,
        ),
]

MagiaLayerNormFp16Bindings = [
    NodeBinding(
        LayerNormChecker(
            [PointerClass(float16_t), PointerClass(float16_t), PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        LayerNormFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    )
]

MagiaLeakyReluFp16Bindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        LeakyReluFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    )
]

MagiaMatMulFp16Bindings = [
    NodeBinding(
        MatMulChecker(
            [PointerClass(float16_t), PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        MatMulFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    )
]

MagiaMaxPool2DFp16Bindings = [
    NodeBinding(
        MaxPoolChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        MaxPool2DFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    )
]

MagiaMulFp16Bindings = [
    NodeBinding(
        MulChecker(
            [PointerClass(float16_t), PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        MulFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    )
]

MagiaReluFp16Bindings = [
    NodeBinding(
        ReluChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        ReluFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    ),
]

MagiaReshapeFp16Bindings = [
    NodeBinding(
        ReshapeChecker(
            [PointerClass(float16_t), PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        ReshapeFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    )
]

MagiaResizeFp16Bindings = [
    NodeBinding(
        PassThroughTypeChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        ResizeFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    )
]

MagiaScatterFp16Bindings = [
    NodeBinding(
        PassThroughTypeChecker(
            [PointerClass(float16_t), PointerClass(int64_t), PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        ScatterFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    )
]

MagiaSeluFp16Bindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        SeluFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    )
]

MagiaSigmoidFp16Bindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        SigmoidFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    ),
]

MagiaSliceFp16Bindings = [
    NodeBinding(
        SliceChecker(
            [PointerClass(float16_t), PointerClass(int64_t), PointerClass(int64_t), PointerClass(int64_t), PointerClass(int64_t)],
            [PointerClass(float16_t)]
        ),
        SliceFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    ),
]

MagiaSoftmaxFp16Bindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        SoftMaxFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    ),
]

MagiaSubFp16Bindings = [
    NodeBinding(
        AddChecker(
            [PointerClass(float16_t), PointerClass(float16_t)],
            [PointerClass(float16_t)]
        ),
        SubFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    ),
]

MagiaSwishFp16Bindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)],
        ),
        SwishFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    ),
]

MagiaTransposeFp16Bindings = [
    NodeBinding(
        TransposeChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)],
        ),
        TransposeFP16SpatzTemplate.referenceTemplate,
        BasicTransformer,
    ),
]
