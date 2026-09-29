# Copyright 2024-2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
# Philip Wiese <wiesep@iis.ee.ethz.ch>
# Alberto Dequino <alberto.dequino@unibo.it>

# Define mappings for tests
if(SPATZ_TESTS AND PULP_TESTS)
    set(TEST_MAPPINGS
        magia:magia,spatz_pulp_on_magia
    )
elseif(SPATZ_TESTS)
    set(TEST_MAPPINGS
        magia:magia,spatz_on_magia
    )
elseif(PULP_TESTS)
    set(TEST_MAPPINGS
        magia:magia,pulp_on_magia
    )
else()
    set(TEST_MAPPINGS
        magia:magia
    )
endif()

# Call the macro
add_magia_subdirectories(${TARGET_PLATFORM} "Test" "${TEST_MAPPINGS}")
