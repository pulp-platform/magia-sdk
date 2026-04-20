# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Philip Wiese <wiesep@iis.ee.ethz.ch>
# Alberto Dequino <alberto.dequino@unibo.it>

# Define mappings for devices
set(DEVICE_MAPPINGS
    magia_v1:
    magia_v2:
)

# Call the macro
add_magia_subdirectories(${TARGET_PLATFORM} "Device" "${DEVICE_MAPPINGS}")