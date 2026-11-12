# Copyright 2026 Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""The L1 size of the SDK's tile address map, for the golden generators that size their blocks over L1."""

import os
import re

# magia-sdk: $MAGIA_SDK (as for mgcc), or the checkout of this file
TILE_ADDR_MAP = os.path.join('targets', 'magia', 'include', 'addr_map', 'tile_addr_map.h')


def l1_bytes():
    """L1_SIZE + 1 of the tile address map: the bytes of L1 the tests may use (the stack and .tile_bss are below
    L1_BASE), the same as mgcc's L1 fit check."""
    sdk = os.environ.get('MAGIA_SDK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
    with open(os.path.join(sdk, TILE_ADDR_MAP)) as f:
        m = re.search(r'^\s*#define\s+L1_SIZE\s+\(?\s*(0x[0-9a-fA-F]+|\d+)\s*\)?', f.read(), re.M)
    return int(m.group(1), 0) + 1


# rows of the M tiles of RedMulE, as mgc's devices.REDMULE_M_TILE: mgc's L1 layout rounds the rows of every buffer
# RedMulE writes up to a multiple of it (the store of the last M tile also writes zeros past the end of Z)
REDMULE_M_TILE = 16


def z_rows(rows):
    """Rows that a RedMulE output of `rows` rows takes in mgc's L1 layout."""
    return -(-rows // REDMULE_M_TILE) * REDMULE_M_TILE
