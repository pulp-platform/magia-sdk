# Copyright 2024-2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
# Alberto Dequino <alberto.dequino@unibo.it>

set(CMAKE_EXECUTABLE_SUFFIX ".elf")

set(CMAKE_SYSTEM_NAME Generic)

set(CROSS_COMPILE       riscv32-unknown-elf)
set(CMAKE_C_COMPILER    ${CROSS_COMPILE}-gcc)
set(CMAKE_CXX_COMPILER  ${CROSS_COMPILE}-g++)
set(CMAKE_ASM_COMPILER  ${CMAKE_C_COMPILER})
set(CMAKE_OBJCOPY       ${CROSS_COMPILE}-objcopy)
set(CMAKE_OBJDUMP       ${CROSS_COMPILE}-objdump)
set(CMAKE_AR            ${CROSS_COMPILE}-ar)
set(CMAKE_STRIP         ${CROSS_COMPILE}-strip)

