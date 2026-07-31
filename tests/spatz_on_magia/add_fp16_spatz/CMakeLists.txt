# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Test of the kernels/spatz_fp16/add library kernel (MAGIA_add_fp16_spatz).
#
# TEST_NAME has to be the kernel's task name: add_spatz_task() derives the
# generated header (add_fp16_spatz_task_bin.h) and the task entry point macro
# (ADD_FP16_SPATZ_TASK) from it, and the kernel source includes/uses both.
set(TEST_NAME add_fp16_spatz)

set(KERNEL_DIR ${CMAKE_SOURCE_DIR}/kernels/spatz_fp16/add)

# Compile the kernel's Spatz task
add_spatz_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES
        ${KERNEL_DIR}/spatz_task/add_fp16_spatz_task.c
        ${KERNEL_DIR}/spatz_task/add_bcast_fp16_spatz_task.c
    FIRST_TASK_NAME add_fp16_spatz_task
    INCLUDE_DIRS
        ${KERNEL_DIR}/include
)

# Compile CV32 executable (test + kernel host side) with embedded Spatz binary
add_cv32_executable_with_spatz(
    TARGET_NAME ${TEST_NAME}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/main.c
        ${KERNEL_DIR}/src/add_fp16_spatz.c
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/test_data
        ${KERNEL_DIR}/include
)
