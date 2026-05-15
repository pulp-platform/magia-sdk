# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0


from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import float16_t, int8_t, int32_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.Targets.Generic.TypeCheckers import AddChecker, DivChecker, GELUChecker, ReluChecker, BatchNormChecker
from MagiaDeeployTarget.Templates import AddTemplate
from MagiaDeeployTarget.Templates import AddFP16SpatzTemplate
from MagiaDeeployTarget.Templates import DivFP16SpatzTemplate
from MagiaDeeployTarget.Templates import GeluFP16SpatzTemplate
from MagiaDeeployTarget.Templates import ReluFP16SpatzTemplate
from MagiaDeeployTarget.Templates import BatchNormFP16SpatzTemplate

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
