# Copyright 2024-2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>

file(GLOB_RECURSE SOURCES
  "src/*"
)

add_library(mglib STATIC ${SOURCES})

# Optimize and enable LTO so the thin wrappers fold into their callers instead of
# each paying a full out-of-line stack frame. -ffat-lto-objects keeps libmglib.a
# usable by non-LTO consumers. (The toolchain sets no default -O, so without this
# mglib builds at -O0.)
target_compile_options(mglib PRIVATE -O2 -flto -ffat-lto-objects)

# Keep the profiling hooks (stnl_*_f) alive in the inlined wait copies pulled in
# via eventunit32.h; these defines otherwise only reach the `runtime` target,
# which mglib does not link.
target_compile_options(mglib PRIVATE
  -DPROFILE_CMP=${PROFILE_CMP}
  -DPROFILE_CMI=${PROFILE_CMI}
  -DPROFILE_CMO=${PROFILE_CMO}
  -DPROFILE_SNC=${PROFILE_SNC}
)

target_include_directories(mglib
  PUBLIC
  "include"
  "${CMAKE_SOURCE_DIR}/targets/${TARGET_PLATFORM}/include"
  "${CMAKE_SOURCE_DIR}/drivers/eventunit32/include"
)

target_link_libraries(mglib
  PUBLIC
    "hal"
)