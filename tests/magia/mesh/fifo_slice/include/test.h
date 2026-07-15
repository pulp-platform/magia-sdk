// Copyright 2025 University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Sizing constants for the fifo_slice test. No golden data: the test builds a
// deterministic per-element pattern at runtime and checks the received payload
// against a recomputed reference (idma_nd style).

#ifndef FIFO_SLICE_TEST_H
#define FIFO_SLICE_TEST_H

// Number of message-producing cases (excludes the producer-only negative case).
#define FIFO_SLICE_NUM_CASES 8

// Max packed payload across cases is 64 B (4x8 fp16); round up generously.
#define FIFO_SLICE_SLOT_BYTES 256u

// Producer-local source buffer; large enough for the widest strided extent.
#define FIFO_SLICE_SRC_BUF_BYTES 512u

#define FIFO_SLICE_PRODUCER 1u
#define FIFO_SLICE_CONSUMER 0u

#endif // FIFO_SLICE_TEST_H
