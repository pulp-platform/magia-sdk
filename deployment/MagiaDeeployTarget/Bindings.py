# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0


from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import float16_t, int8_t, int32_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.Targets.Generic.TypeCheckers import DummyChecker, AddChecker, DivChecker, GELUChecker, ReluChecker, BatchNormChecker, GEMMChecker, MaxPoolChecker, LayerNormChecker
from MagiaDeeployTarget.Templates import AddTemplate
from MagiaDeeployTarget.Templates import AddFP16SpatzTemplate
from MagiaDeeployTarget.Templates import DivFP16SpatzTemplate
from MagiaDeeployTarget.Templates import GeluFP16SpatzTemplate
from MagiaDeeployTarget.Templates import ReluFP16SpatzTemplate
from MagiaDeeployTarget.Templates import BatchNormFP16SpatzTemplate
from MagiaDeeployTarget.Templates import GemmFP16SpatzTemplate
from MagiaDeeployTarget.Templates import MaxPool2DFP16SpatzTemplate
from MagiaDeeployTarget.Templates import LayerNormFP16SpatzTemplate
from MagiaDeeployTarget.Templates import CeilFP16SpatzTemplate

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

MagiaCeilFp16Bindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float16_t)],
            [PointerClass(float16_t)]),
            CeilFP16SpatzTemplate.referenceTemplate,
            BasicTransformer,
        )
]
