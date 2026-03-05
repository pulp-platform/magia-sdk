# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0

# WIESEP: It is important to set the ISA and ABI for the host and the cluster snitch
set(ABI_MESH ilp32)
set(ISA_MESH rv32imcxgap9)
set(PICOLIB_MESH rv32im/ilp32)
set(COMPILERRT_MESH rv32imcxgap9)

set(ABI_CLUSTER_SNITCH None)
set(ISA_CLUSTER_SNITCH None)
set(PICOLIB_CLUSTER_SNITCH None)
set(COMPILERRT_CLUSTER_SNITCH None)