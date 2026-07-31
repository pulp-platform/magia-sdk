# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Whole-network TinyViT-5M. Unlike the single-op tests this needs nine kernels in one
# binary, so one add_spatz_task takes all of their Spatz tasks (ten of them - add
# contributes both its elementwise and its broadcast entry point); the generated header
# is named after TEST_NAME, and shim/ redirects each kernel's own <op>_task_bin.h include
# at it (see the comment in any of those shims).
set(TEST_NAME onnx_tinyvit5m)

set(KERNELS_DIR ${CMAKE_SOURCE_DIR}/kernels/spatz_fp16)

set(TVIT_KERNELS
    conv2dgemm
    gelu
    layernorm
    matmul
    gemm
    add
    softmax
    transpose
    globalaveragepool
)

set(TVIT_TASK_SOURCES "")
set(TVIT_KERNEL_SOURCES "")
set(TVIT_KERNEL_INCLUDES "")
foreach(K ${TVIT_KERNELS})
    list(APPEND TVIT_TASK_SOURCES   ${KERNELS_DIR}/${K}/spatz_task/${K}_fp16_spatz_task.c)
    list(APPEND TVIT_KERNEL_SOURCES ${KERNELS_DIR}/${K}/src/${K}_fp16_spatz.c)
    list(APPEND TVIT_KERNEL_INCLUDES ${KERNELS_DIR}/${K}/include)
endforeach()

# 38 of the 72 Adds broadcast a bias over the last axis, which is a second task in the
# same kernel directory.
list(APPEND TVIT_TASK_SOURCES ${KERNELS_DIR}/add/spatz_task/add_bcast_fp16_spatz_task.c)

add_spatz_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES ${TVIT_TASK_SOURCES}
    FIRST_TASK_NAME conv2dgemm_fp16_spatz_task
    INCLUDE_DIRS ${TVIT_KERNEL_INCLUDES}
)

# The assembler's own dependency scan does not see through .incbin, so regenerating the
# blob would otherwise leave a stale object linked in - the goldens are in there, and a
# stale one fails as a wrong result rather than as a build error.
set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/tinyvit_data.S
             APPEND PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_data/tinyvit_data.bin)

# test_data is on the path for the assembler too: tinyvit_data.S resolves its
# .incbin of tinyvit_data.bin through -I.
add_cv32_executable_with_spatz(
    TARGET_NAME ${TEST_NAME}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/main.c
        ${CMAKE_CURRENT_SOURCE_DIR}/tinyvit_data.S
        ${TVIT_KERNEL_SOURCES}
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/shim
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/test_data
        ${TVIT_KERNEL_INCLUDES}
)

# Stop after the first N layers, for bringing the test up against the per-layer tensors
# in test_data/layers/ instead of waiting on a full inference:
#   cmake -DTVIT_MAX_LAYERS=3 ... && cmake --build build --target onnx_tinyvit5m
if(TVIT_MAX_LAYERS)
    target_compile_definitions(${TEST_NAME} PRIVATE TVIT_MAX_LAYERS=${TVIT_MAX_LAYERS})
endif()

# Per-phase cycle counters in conv2dgemm plus a *per-layer* compute/barrier split here.
# Off by default: it costs a printf per layer per tile. The per-inference totals are
# printed unconditionally (two CSR reads per layer), so this is only needed to find which
# layer the cycles went to. Both only work on GVSoC.
#   cmake -DCONV2DGEMM_PROFILE=1 -DTVIT_MAX_LAYERS=6 ...
if(CONV2DGEMM_PROFILE)
    target_compile_definitions(${TEST_NAME} PRIVATE CONV2DGEMM_PROFILE=1)
endif()
