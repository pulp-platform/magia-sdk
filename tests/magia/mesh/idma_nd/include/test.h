// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Test for the generalized md->nd IDMA data mover (idma_memcpy_md_to_nd).

#ifndef _TEST_IDMA_ND_INCLUDE_GUARD_
#define _TEST_IDMA_ND_INCLUDE_GUARD_

// Byte size of each per-side working buffer (L2 scratch and L1 scratch). Every
// case must keep its addressed extent (including per-dim start offsets) below it.
#define IDMA_ND_BUF_BYTES 1024u

#endif
