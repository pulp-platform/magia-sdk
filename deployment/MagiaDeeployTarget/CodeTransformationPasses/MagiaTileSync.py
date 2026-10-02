# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, _NoVerbosity

magiaSyncTilesFunctionTemplate = NodeTemplate("""
static inline void magia_sync_tiles() {
    fsync_sync_global();
    eu_fsync_wait(WFE);
}
""")

_syncTilesTemplate = NodeTemplate("""
magia_sync_tiles();
""")


class MagiaSynchTilesPass(CodeTransformationPass):

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        executionBlock.addRight(_syncTilesTemplate, {})
        return ctxt, executionBlock
