# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

set(TEST_NAME onnx_rope)

set(KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../kernels/rope/fp16/spatz)
set(KERNEL_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../kernels/common)

# Compile Spatz task
add_spatz_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES ${KERNEL_DIR}/spatz_task/rope_fp16_spatz_task.c
    FIRST_TASK_NAME rope_fp16_spatz_task
    INCLUDE_DIRS
        ${KERNEL_COMMON_DIR}
        ${KERNEL_DIR}/include
)

# Compile CV32 executable with embedded Spatz binary
add_cv32_executable_with_spatz(
    TARGET_NAME ${TEST_NAME}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/main.c
        ${KERNEL_DIR}/src/rope_fp16_spatz.c
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/test_data
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${KERNEL_DIR}/include
        ${KERNEL_COMMON_DIR}
)
