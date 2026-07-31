# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Whole-network MobileViT-v2. Unlike the single-op tests this needs eleven kernels in one
# binary, so one add_spatz_task takes all of their Spatz tasks (twelve of them - mul
# contributes both its elementwise and its broadcast entry point); the generated header is
# named after TEST_NAME, and shim/ redirects each kernel's own <op>_task_bin.h include at
# it (see the comment in any of those shims).
set(TEST_NAME onnx_mobilevit_v2)

set(KERNELS_DIR ${CMAKE_SOURCE_DIR}/kernels/spatz_fp16)

set(MVIT_KERNELS
    conv2dgemm
    relu
    add
    mul
    sigmoid
    groupnorm
    softmax
    reducesum
    transpose
    globalaveragepool
    gemm
)

set(MVIT_TASK_SOURCES "")
set(MVIT_KERNEL_SOURCES "")
set(MVIT_KERNEL_INCLUDES "")
foreach(K ${MVIT_KERNELS})
    list(APPEND MVIT_TASK_SOURCES   ${KERNELS_DIR}/${K}/spatz_task/${K}_fp16_spatz_task.c)
    list(APPEND MVIT_KERNEL_SOURCES ${KERNELS_DIR}/${K}/src/${K}_fp16_spatz.c)
    list(APPEND MVIT_KERNEL_INCLUDES ${KERNELS_DIR}/${K}/include)
endforeach()

# 18 of the 43 Muls broadcast, which is a second task in the same kernel directory.
list(APPEND MVIT_TASK_SOURCES ${KERNELS_DIR}/mul/spatz_task/mul_bcast_fp16_spatz_task.c)

# No Add here broadcasts, but add/src references ADD_BCAST_FP16_SPATZ_TASK unconditionally
# the way mul/src does, so the task has to be in this binary's table to link.
list(APPEND MVIT_TASK_SOURCES ${KERNELS_DIR}/add/spatz_task/add_bcast_fp16_spatz_task.c)

add_spatz_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES ${MVIT_TASK_SOURCES}
    FIRST_TASK_NAME conv2dgemm_fp16_spatz_task
    INCLUDE_DIRS ${MVIT_KERNEL_INCLUDES}
)

# The assembler's own dependency scan does not see through .incbin, so regenerating the
# blob would otherwise leave a stale object linked in - the goldens are in there, and a
# stale one fails as a wrong result rather than as a build error.
set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/mobilevit_data.S
             APPEND PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_data/mobilevit_data.bin)

# test_data is on the path for the assembler too: mobilevit_data.S resolves its
# .incbin of mobilevit_data.bin through -I.
add_cv32_executable_with_spatz(
    TARGET_NAME ${TEST_NAME}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/main.c
        ${CMAKE_CURRENT_SOURCE_DIR}/mobilevit_data.S
        ${MVIT_KERNEL_SOURCES}
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/shim
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test_data
        ${MVIT_KERNEL_INCLUDES}
)

# Stop after the first N layers, for bringing the test up against the per-layer tensors
# in test_data/layers/ instead of waiting on a full inference:
#   cmake -DMVIT_MAX_LAYERS=3 ... && cmake --build build --target onnx_mobilevit_v2
if(MVIT_MAX_LAYERS)
    target_compile_definitions(${TEST_NAME} PRIVATE MVIT_MAX_LAYERS=${MVIT_MAX_LAYERS})
endif()

# Per-phase cycle counters in conv2dgemm plus the per-layer compute/barrier split here.
# Off by default: it costs a printf per layer per tile and only works on GVSoC.
#   cmake -DCONV2DGEMM_PROFILE=1 -DMVIT_MAX_LAYERS=6 ...
if(CONV2DGEMM_PROFILE)
    target_compile_definitions(${TEST_NAME} PRIVATE CONV2DGEMM_PROFILE=1)
endif()
