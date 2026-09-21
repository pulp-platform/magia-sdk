# Copyright 2026 Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

set(TEST_NAME test_idma_3d_benchmark)

add_executable(${TEST_NAME} src/test.c)
target_compile_options(${TEST_NAME} PRIVATE -O2)
target_link_libraries(${TEST_NAME} PUBLIC runtime hal)
