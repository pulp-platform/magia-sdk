# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

# Declare every buffer as a statically-allocated array instead of an
# unallocated pointer: there is no L2 allocator in the SDK (magia_l2_malloc is
# not implemented), so pointer buffers would stay NULL and the first kernel
# writing its output would dereference it.
# Intermediate buffers are emitted inside RunNetwork, so they must be 'static'
# (goes to .bss) or they would blow the small per-tile stack. I/O buffers are
# emitted at file scope and are 'extern'-ed in network.h, so they stay
# non-static (external linkage).
# Aliased buffers (e.g. a Reshape output) share the producer's storage. Deeploy
# expresses this as `out = in`, but here buffers are ARRAYS, not pointers (no
# allocator yet), and a C array can't be reassigned. So an alias is emitted as a
# pointer to the source array instead. Once MAGIA gets an allocator and buffers
# become pointers, this can go back to the plain `out = in`.
magiaInitTemplate = NodeTemplate("""\
% if alias is not None:
${type.referencedType.typeName} *${name} = ${alias};
% elif is_io:
${type.referencedType.typeName} ${name}[${size}];
% else:
static ${type.referencedType.typeName} ${name}[${size}];
% endif
""")

# Buffers are statically allocated by magiaInitTemplate, so there is nothing to
# allocate at runtime.
magiaAllocateTemplate = NodeTemplate("")

magiaGlobalInitTemplate = NodeTemplate("""
% if _memoryLevel == "L1":
static ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n
% elif _memoryLevel == "L2" or _memoryLevel is None:
extern ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n
% endif
""")

magiaGlobalAllocateTemplate = NodeTemplate("")