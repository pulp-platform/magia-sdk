# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <f.conti@unibo.it>
"""The mgc pass pipeline, in order (see ../../README.md)."""

from .events import EventAlloc
from .job_lower import JobLower
from .l1_layout import L1Layout
from .legalize_dma import LegalizeDma
from .multibuffer import MultiBuffer
from .pipeline import Pipeline

PASSES = [LegalizeDma(), Pipeline(), JobLower(), MultiBuffer(), L1Layout(), EventAlloc()]
