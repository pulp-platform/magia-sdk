# Copyright 2025-2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Alberto Dequino <alberto.dequino@unibo.it>

# Add test
add_subdirectory(mesh)

add_custom_target(test_maps_operation_indexing ALL
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run-maps-operation-indexing.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Running host MAPS operation indexing contract test")
