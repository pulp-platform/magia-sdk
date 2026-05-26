# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0


from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import float16_t, int8_t, int32_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.Targets.Generic.TypeCheckers import AddChecker, BatchNormChecker, DivChecker, DummyChecker, GELUChecker, GEMMChecker, LayerNormChecker, MaxPoolChecker, ReluChecker
from MagiaDeeployTarget.Templates import AddFP16SpatzTemplate
from MagiaDeeployTarget.Templates import AddTemplate
from MagiaDeeployTarget.Templates import AveragePool2DFP16SpatzTemplate
from MagiaDeeployTarget.Templates import BatchNormFP16SpatzTemplate
from MagiaDeeployTarget.Templates import CeilFP16SpatzTemplate
from MagiaDeeployTarget.Templates import ClipFP16SpatzTemplate
from MagiaDeeployTarget.Templates import DivFP16SpatzTemplate
from MagiaDeeployTarget.Templates import ExpFP16SpatzTemplate
from MagiaDeeployTarget.Templates import FloorFP16SpatzTemplate
from MagiaDeeployTarget.Templates import GeluFP16SpatzTemplate
from MagiaDeeployTarget.Templates import GemmFP16SpatzTemplate
from MagiaDeeployTarget.Templates import GlobalAveragePoolFP16SpatzTemplate
from MagiaDeeployTarget.Templates import GlobalMaxPoolFP16SpatzTemplate
from MagiaDeeployTarget.Templates import GroupNormFP16SpatzTemplate
from MagiaDeeployTarget.Templates import InstanceNormFP16SpatzTemplate
from MagiaDeeployTarget.Templates import LayerNormFP16SpatzTemplate
from MagiaDeeployTarget.Templates import MaxPool2DFP16SpatzTemplate
from MagiaDeeployTarget.Templates import ReluFP16SpatzTemplate
from MagiaDeeployTarget.Templates import SigmoidFP16SpatzTemplate
from MagiaDeeployTarget.Templates import SubFP16SpatzTemplate

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
