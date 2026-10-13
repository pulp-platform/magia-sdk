/*
 * Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ResNet18's weights, input image and goldens, as one FP16 blob in L2.
 *
 * .l2_bulk lives at 0xc0000000, in the 192 MB of L2 below the program image (see
 * targets/magia_v3/link.ld). It has to go there because dataram ends just under
 * TEST_END_ADDR and cannot grow to hold 22.6 MB.
 *
 * .incbin rather than a generated C array: 11.7 M FP16 values would be a ~180 MB C
 * source, which is not something to hand a compiler. The assembler resolves the path
 * through -I, so test_data has to be on the include path (the CMakeLists puts it there).
 * Regenerate the blob with: cd test_data && python3 generator.py
 */

    .section .l2_bulk, "a", @progbits
    .globl resnet18_data
    .balign 4
resnet18_data:
    .incbin "resnet18_data.bin"
    .globl resnet18_data_end
resnet18_data_end:
    .size resnet18_data, . - resnet18_data
