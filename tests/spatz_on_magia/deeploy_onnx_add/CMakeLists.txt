# Copyright 2026 Fondazione ChipsIT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# 
# Alex Marchioni <alex.marchioni@chips.it>

set(TEST_NAME deeploy_onnx_add)

# Step 1: Compile the Spatz task and generate the C header
add_spatz_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../../kernels/onnx_add/spatz_task/onnx_add_task.c
    FIRST_TASK_NAME onnx_add_task
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../kernels/onnx_add/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Step 2: Compile the CV32 executable and embed the Spatz binary
add_cv32_executable_with_spatz(
    TARGET_NAME ${TEST_NAME}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../../kernels/onnx_add/src/onnx_add.c src/network.c src/main.c src/data.c
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../kernels/onnx_add/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../kernels/onnx_add/../common
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

