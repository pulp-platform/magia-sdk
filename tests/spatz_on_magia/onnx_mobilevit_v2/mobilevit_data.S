/*
 * Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 *
 * MobileViT-v2's weights, input image and goldens, as one FP16 blob in L2.
 *
 * .l2_bulk lives at 0xc0000000, in the 192 MB of L2 below the program image (see
 * targets/magia_v3/link.ld). The blob itself is only 2.8 MB and would fit dataram, but
 * the 39.4 MB activation arena next to it does not, and keeping the two together is
 * what makes the offsets in mobilevit_graph.h mean one thing.
 *
 * .incbin rather than a generated C array: the assembler resolves the path through -I,
 * so test_data has to be on the include path (the CMakeLists puts it there).
 * Regenerate the blob with: cd test_data && python3 generator.py
 */

    .section .l2_bulk, "a", @progbits
    .globl mobilevit_data
    .balign 4
mobilevit_data:
    .incbin "mobilevit_data.bin"
    .globl mobilevit_data_end
mobilevit_data_end:
    .size mobilevit_data, . - mobilevit_data
