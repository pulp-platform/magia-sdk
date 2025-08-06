// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>
// Victor Isachi <victor.isachi@unibo.it>

#ifndef _TEST_FLATATT_INCLUDE_GUARD_
#define _TEST_FLATATT_INCLUDE_GUARD_

#define S_SIZE (512)
#define D_SIZE (128)

extern _Float16 q_inp       [S_SIZE * D_SIZE] = {};
extern _Float16 k_inp       [D_SIZE * S_SIZE] = {};
extern _Float16 v_inp       [S_SIZE * D_SIZE] = {};
extern _Float16 o_out       [S_SIZE * D_SIZE] = {};
extern _Float16 o_golden    [S_SIZE * D_SIZE] = {};

#endif //_TEST_FLATATT_INCLUDE_GUARD_

