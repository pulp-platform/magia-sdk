# Copyright 2024-2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Viviane Potocnik <vivianep@iis.ee.ethz.ch>
# Alberto Dequino <alberto.dequino@unibo.it>

include(ExternalProject)

message(STATUS "[MAGIA-SDK] Setting up picolibc for Host (ISA: ${ISA}, ABI: ${ABI})")

set(CROSS_C_COMPILER "${CMAKE_C_COMPILER}")
set(CROSS_C_COMPILER_ARGS "-target ${CROSS_COMPILE} -march=${ISA} -nostdlib" CACHE STRING "Compiler arguments")
# VIVIANEP: These flags are only for building Picolibc; adding them globally breaks
#           app builds (e.g., missing <sys/types.h>) or causes issues in freestanding mode.
set(CROSS_C_ARGS "-Werror=double-promotion -Wno-unsupported-floating-point-opt -fshort-enums ${CMAKE_ALT_C_OPTIONS} -march=${ISA} -mabi=${ABI}")
set(CROSS_C_LINK_ARGS "-Wl,-z,noexecstack -march=${ISA} -mabi=${ABI}")

set(CROSS_AR "${CMAKE_AR}")
set(CROSS_STRIP "${CMAKE_STRIP}")

set(CROSS_SKIP_SANITY_CHECK "true")

# Prepare Meson arrays
function(prepare_meson_array output_var input_string)
    string(REPLACE " " ";" temp_list "${input_string}")
    set(formatted_list "")
    foreach(item IN LISTS temp_list)
        list(APPEND formatted_list "'${item}'")
    endforeach()
    string(JOIN ", " result ${formatted_list})
    set(${output_var} "${result}" PARENT_SCOPE)
endfunction()

prepare_meson_array(CROSS_C_COMPILER_ARGS_LIST "${CROSS_C_COMPILER_ARGS}")
prepare_meson_array(CROSS_C_ARGS_LIST "${CROSS_C_ARGS}")
prepare_meson_array(CROSS_C_LINK_ARGS_LIST "${CROSS_C_LINK_ARGS}")

set(PICOLIBC_SRC_DIR ${CMAKE_BINARY_DIR}/picolibc-src)

set(PICOLIBC_BUILD_DIR ${CMAKE_BINARY_DIR}/picolibc-build-${ISA}-${ABI})
set(PICOLIBC_INSTALL_DIR ${CMAKE_BINARY_DIR}/picolibc-install-${ISA}-${ABI})
set(PICOLIBC_CROSS_FILE ${CMAKE_BINARY_DIR}/picolibc-cross-file-${ISA}-${ABI}.txt)


# Generate the Meson cross-file
configure_file(${CMAKE_CURRENT_LIST_DIR}/../scripts/picolibc-cross-file.txt.in ${PICOLIBC_CROSS_FILE} @ONLY)

message(STATUS "[MAGIA-SDK] Saving cross compilation file to ${PICOLIBC_CROSS_FILE}")
# Add picolibc as an external project
ExternalProject_Add(
    picolibc-${ISA}-${ABI}
    GIT_REPOSITORY https://github.com/picolibc/picolibc.git
    GIT_TAG main
    SOURCE_DIR ${PICOLIBC_SRC_DIR}
    BINARY_DIR ${PICOLIBC_BUILD_DIR}
    INSTALL_DIR ${PICOLIBC_INSTALL_DIR}
    CONFIGURE_COMMAND meson setup ${PICOLIBC_BUILD_DIR} ${PICOLIBC_SRC_DIR} --cross-file ${PICOLIBC_CROSS_FILE} --prefix ${PICOLIBC_INSTALL_DIR} --default-library=static
    BUILD_COMMAND ninja -C ${PICOLIBC_BUILD_DIR}
    INSTALL_COMMAND ninja -C ${PICOLIBC_BUILD_DIR} install
    BUILD_BYPRODUCTS ${PICOLIBC_INSTALL_DIR}/lib/libc.a
    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
)

set(PICOLIBC_TARGET picolibc-${ISA}-${ABI})

add_library(picolibc STATIC IMPORTED GLOBAL)

set_target_properties(picolibc PROPERTIES
    IMPORTED_LOCATION "${PICOLIBC_INSTALL_DIR}/lib/libc.a"
)

add_dependencies(picolibc picolibc-${ISA_HOST}-${ABI})