# Copyright 2026 Fondazione ChipsIT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Alex Marchioni <alex.marchioni@chips.it>

file(GLOB_RECURSE SOURCES
    src/*.c
)

add_library(kernels STATIC ${SOURCES})

target_include_directories(kernels PUBLIC "include")

target_link_libraries(kernels
  PUBLIC
    runtime
    hal
)