// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Sizing constants for the fifo_ring circular-buffer test. No golden data.

#ifndef FIFO_RING_TEST_H
#define FIFO_RING_TEST_H

#define FIFO_RING_CONSUMER 0u

// Each producer streams MSGS_PER_PROD messages through a sub-ring of only
// SLOTS_PER_PROD slots, so head/tail wrap the sub-ring several times and the
// producer repeatedly hits backpressure (fifo_reserve spinning while full).
#define FIFO_RING_MSGS_PER_PROD  4u
#define FIFO_RING_SLOTS_PER_PROD 2u

#define FIFO_RING_ROWS       1u  // rows per message block
#define FIFO_RING_COLS       8u  // columns per message block
#define FIFO_RING_ELEM_BYTES 2u  // fp16-sized elements
#define FIFO_RING_SLOT_BYTES 64u // >= ROWS*COLS*ELEM_BYTES

// Upper bound on tiles (16x16 mesh) for the consumer's bookkeeping array.
#define FIFO_RING_MAX_TILES 256u

#endif // FIFO_RING_TEST_H
