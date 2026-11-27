# Copyright 2025-2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Alberto Dequino <alberto.dequino@unibo.it>

# Add test
add_subdirectory(mesh)
# The mgc tests have goldens for 2x2 to 16x16 meshes only (#error in their test.h otherwise)
if(NOT TILES STREQUAL "1")
    add_subdirectory(mgc)
endif()
