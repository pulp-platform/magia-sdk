// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>

#ifndef _TEST_FSYNC_LEVELS_INCLUDE_GUARD_
#define _TEST_FSYNC_LEVELS_INCLUDE_GUARD_

int write_delayed(uint8_t lvl, uint32_t id, uint8_t groupid, uint32_t addr);

int check_values(uint8_t lvl, uint8_t groupid, uint32_t addr, uint8_t dir);

#endif //_TEST_FSYNC_LEVELS_INCLUDE_GUARD_