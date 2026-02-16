// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>

#ifndef _TEST_AMO_LOCK_INCLUDE_GUARD_
#define _TEST_AMO_LOCK_INCLUDE_GUARD_

#include "utils/amo_utils.h"

extern uint32_t lock = 0x12345678;
extern uint32_t value = NUM_HARTS;

#endif //_TEST_AMO_LOCK_INCLUDE_GUARD_