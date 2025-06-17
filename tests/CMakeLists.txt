# Copyright 2024-2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
# Philip Wiese <wiesep@iis.ee.ethz.ch>
# Alberto Dequino <alberto.dequino@unibo.it>

# Define mappings for tests
set(TEST_MAPPINGS
    magia:magia
)

# Call the macro
add_magia_subdirectories(${TARGET_PLATFORM} "Test" "${TEST_MAPPINGS}")