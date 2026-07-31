# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Test of the kernels/spatz_fp16/gelu library kernel (MAGIA_gelu_fp16_spatz).
#
# Note that tests/spatz_on_magia/onnx_gelu does NOT cover this kernel: it carries its
# own local spatz_task/onnx_gelu_task.c and never links kernels/spatz_fp16/gelu.
#
# TEST_NAME has to be the kernel's task name: add_spatz_task() derives the
# generated header (<name>_task_bin.h) and the task entry macro from it, and the
# kernel source includes/uses both.
set(TEST_NAME gelu_fp16_spatz)

set(KERNEL_DIR ${CMAKE_SOURCE_DIR}/kernels/spatz_fp16/gelu)

add_spatz_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES ${KERNEL_DIR}/spatz_task/gelu_fp16_spatz_task.c
    FIRST_TASK_NAME gelu_fp16_spatz_task
    INCLUDE_DIRS
        ${KERNEL_DIR}/include
)

add_cv32_executable_with_spatz(
    TARGET_NAME ${TEST_NAME}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/main.c
        ${KERNEL_DIR}/src/gelu_fp16_spatz.c
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/test_data
        ${KERNEL_DIR}/include
)
