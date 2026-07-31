# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# TEST_NAME has to be the kernel's task name: add_spatz_task() derives the
# generated header (<name>_task_bin.h) and the task entry macro from it, and the
# kernel source includes/uses both.
set(TEST_NAME transpose_fp16_spatz)

set(KERNEL_DIR ${CMAKE_SOURCE_DIR}/kernels/spatz_fp16/transpose)

add_spatz_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES ${KERNEL_DIR}/spatz_task/transpose_fp16_spatz_task.c
    FIRST_TASK_NAME transpose_fp16_spatz_task
    INCLUDE_DIRS
        ${KERNEL_DIR}/include
)

add_cv32_executable_with_spatz(
    TARGET_NAME ${TEST_NAME}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/main.c
        ${KERNEL_DIR}/src/transpose_fp16_spatz.c
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/test_data
        ${KERNEL_DIR}/include
)
