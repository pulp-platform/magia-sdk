# Copyright 2024-2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
# Alberto Dequino <alberto.dequino@unibo.it>

set(CMAKE_EXECUTABLE_SUFFIX ".elf")

set(CMAKE_SYSTEM_NAME Generic)

set(RISCV_TOOLCHAIN_PREFIX "$ENV{HOME}/riscv/bin/riscv32-unknown-elf")

set(CMAKE_C_COMPILER ${RISCV_TOOLCHAIN_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${RISCV_TOOLCHAIN_PREFIX}-g++)
set(CMAKE_ASM_COMPILER ${CMAKE_C_COMPILER})
set(CMAKE_OBJCOPY ${RISCV_TOOLCHAIN_PREFIX}-objcopy)
set(CMAKE_OBJDUMP ${RISCV_TOOLCHAIN_PREFIX}-objdump)
set(CMAKE_AR ${RISCV_TOOLCHAIN_PREFIX}-ar)
