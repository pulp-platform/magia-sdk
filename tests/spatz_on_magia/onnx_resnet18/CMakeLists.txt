# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Whole-network ResNet18. Unlike the single-op tests this needs seven kernels in one
# binary, so one add_spatz_task takes all seven Spatz tasks; the generated header is
# named after TEST_NAME, and shim/ redirects each kernel's own <op>_task_bin.h include
# at it (see the comment in any of those shims).
set(TEST_NAME onnx_resnet18)

set(KERNELS_DIR ${CMAKE_SOURCE_DIR}/kernels/spatz_fp16)

set(RN18_KERNELS
    conv2dgemm
    relu
    add
    maxpool2d
    globalaveragepool
    gemm
    softmax
)

set(RN18_TASK_SOURCES "")
set(RN18_KERNEL_SOURCES "")
set(RN18_KERNEL_INCLUDES "")
foreach(K ${RN18_KERNELS})
    list(APPEND RN18_TASK_SOURCES   ${KERNELS_DIR}/${K}/spatz_task/${K}_fp16_spatz_task.c)
    list(APPEND RN18_KERNEL_SOURCES ${KERNELS_DIR}/${K}/src/${K}_fp16_spatz.c)
    list(APPEND RN18_KERNEL_INCLUDES ${KERNELS_DIR}/${K}/include)
endforeach()

# No Add here broadcasts, but add/src references ADD_BCAST_FP16_SPATZ_TASK unconditionally
# the way mul/src does, so the task has to be in this binary's table to link.
list(APPEND RN18_TASK_SOURCES ${KERNELS_DIR}/add/spatz_task/add_bcast_fp16_spatz_task.c)

add_spatz_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES ${RN18_TASK_SOURCES}
    FIRST_TASK_NAME conv2dgemm_fp16_spatz_task
    INCLUDE_DIRS ${RN18_KERNEL_INCLUDES}
)

# The assembler's own dependency scan does not see through .incbin, so regenerating the
# blob would otherwise leave a stale object linked in - the goldens are in there, and a
# stale one fails as a wrong result rather than as a build error.
set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/resnet18_data.S
             APPEND PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_data/resnet18_data.bin)

# test_data is on the path for the assembler too: resnet18_data.S resolves its
# .incbin of resnet18_data.bin through -I.
add_cv32_executable_with_spatz(
    TARGET_NAME ${TEST_NAME}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/main.c
        ${CMAKE_CURRENT_SOURCE_DIR}/resnet18_data.S
        ${RN18_KERNEL_SOURCES}
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/shim
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test_data
        ${RN18_KERNEL_INCLUDES}
)

# Stop after the first N layers, for bringing the test up against the per-layer tensors
# in test_data/layers/ instead of waiting on a full inference:
#   cmake -DRN18_MAX_LAYERS=3 ... && cmake --build build --target onnx_resnet18
if(RN18_MAX_LAYERS)
    target_compile_definitions(${TEST_NAME} PRIVATE RN18_MAX_LAYERS=${RN18_MAX_LAYERS})
endif()

# Per-phase cycle counters in conv2dgemm plus the per-layer compute/barrier split here.
# Off by default: it costs a printf per layer per tile and only works on GVSoC.
#   cmake -DCONV2DGEMM_PROFILE=1 -DRN18_MAX_LAYERS=6 ...
if(CONV2DGEMM_PROFILE)
    target_compile_definitions(${TEST_NAME} PRIVATE CONV2DGEMM_PROFILE=1)
endif()
