/*
 * Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 *
 * TinyViT-5M's weights, input image and goldens, as one FP16 blob in L2.
 *
 * .l2_bulk lives at 0xc0000000, in the 192 MB of L2 below the program image (see
 * targets/magia_v3/link.ld). Neither the 12.9 MB blob nor the 59 MB activation arena
 * next to it would fit dataram, and keeping the two together is what makes the offsets
 * in tinyvit_graph.h mean one thing.
 *
 * .incbin rather than a generated C array: 6.6 M values would be a ~100 MB C source,
 * while a two-line .S assembles instantly. The assembler resolves the path through -I,
 * so test_data has to be on the include path (the CMakeLists puts it there).
 * Regenerate the blob with: cd test_data && python3 generator.py
 */

    .section .l2_bulk, "a", @progbits
    .globl tinyvit_data
    .balign 4
tinyvit_data:
    .incbin "tinyvit_data.bin"
    .globl tinyvit_data_end
tinyvit_data_end:
    .size tinyvit_data, . - tinyvit_data
