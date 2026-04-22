# Copyright 2026 Fondazione ChipsIT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# 
# Alex Marchioni <alex.marchioni@chips.it>

set(TEST_NAME test_adder)

file(GLOB_RECURSE TEST_SRCS "src/*.c")

add_executable(${TEST_NAME} ${TEST_SRCS})
target_include_directories(${TEST_NAME} PUBLIC include)

target_compile_options(${TEST_NAME} PRIVATE -O2)
target_link_libraries(${TEST_NAME} PUBLIC runtime hal kernels)

add_custom_command(
    TARGET ${TEST_NAME}
    POST_BUILD
    COMMAND ${CMAKE_OBJDUMP} -dhS -Mmarch=${ISA}  $<TARGET_FILE:${TEST_NAME}> > $<TARGET_FILE:${TEST_NAME}>.s
)
