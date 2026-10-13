# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

set(TEST_NAME group_reductions_fp16_spatz)
set(KERNELS_DIR ${CMAKE_SOURCE_DIR}/kernels)

add_spatz_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES
        ${KERNELS_DIR}/group_reduce/fp16/spatz/spatz_task/group_reduce_fp16_spatz_task.c
        ${KERNELS_DIR}/group_centered_reduce/fp16/spatz/spatz_task/group_centered_reduce_fp16_spatz_task.c
    FIRST_TASK_NAME group_reduce_fp16_spatz_task
    INCLUDE_DIRS
        ${KERNELS_DIR}/group_reduce/fp16/spatz/include
        ${KERNELS_DIR}/group_centered_reduce/fp16/spatz/include
)

add_cv32_executable_with_spatz(
    TARGET_NAME ${TEST_NAME}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.c
    INCLUDE_DIRS
        ${KERNELS_DIR}/group_reduce/fp16/spatz/include
        ${KERNELS_DIR}/group_centered_reduce/fp16/spatz/include
)

# The golden is computed on the CV32, which has no FP16 arithmetic (zhinxmin):
# make every float16 cast and assignment round to FP16 and keep a * b + c
# unfused, so it rounds each step like the Spatz task does.
target_compile_options(${TEST_NAME} PRIVATE -fexcess-precision=standard -ffp-contract=off)
