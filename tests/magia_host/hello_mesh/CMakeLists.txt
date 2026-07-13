# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

set(TEST_NAME hello_mesh)

# Compile Mesh task
add_mesh_task(
    TEST_NAME ${TEST_NAME}
    TASK_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/mesh_task/hello_task.c
    FIRST_TASK_NAME hello_task
)

# Compile CV32 Host executable with embedded Mesh binary
add_cv32_executable_with_mesh(
    TARGET_NAME ${TEST_NAME}
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.c
)
