// Copyright 2025-2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>

#ifndef _TEST_AMO_LOCK_GLOBAL_INCLUDE_GUARD_
#define _TEST_AMO_LOCK_GLOBAL_INCLUDE_GUARD_

#include "utils/amo_utils.h"

uint32_t lock  = 0x12345678;
uint32_t value = NUM_HARTS;

#endif //_TEST_AMO_LOCK_GLOBAL_INCLUDE_GUARD_