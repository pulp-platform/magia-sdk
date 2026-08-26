# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

set(TEST_NAME group_normalize_fp16_spatz)
set(KERNEL_DIR ${CMAKE_SOURCE_DIR}/kernels/spatz_fp16/group_normalize)

add_spatz_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES ${KERNEL_DIR}/spatz_task/group_normalize_fp16_spatz_task.c
    FIRST_TASK_NAME group_normalize_fp16_spatz_task
    INCLUDE_DIRS ${KERNEL_DIR}/include
)

add_cv32_executable_with_spatz(
    TARGET_NAME ${TEST_NAME}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.c
    INCLUDE_DIRS ${KERNEL_DIR}/include
)
