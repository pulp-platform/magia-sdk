// Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Francesco Conti <f.conti@unibo.it>

#pragma once

// Input of the nearest x2 upsample: UP_H x UP_W pixels of UP_C fp16 channels, HWC (YOLO26x neck: 768 ch, 20x20 and
// 40x40). Overridable with -D.
#ifndef UP_C
#define UP_C 768
#endif
#ifndef UP_H
#define UP_H 20
#endif
#ifndef UP_W
#define UP_W 20
#endif
