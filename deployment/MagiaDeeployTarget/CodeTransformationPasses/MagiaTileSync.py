# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, _NoVerbosity

magiaSyncTilesFunctionTemplate = NodeTemplate("""
static void magia_sync_tiles(void) {
    fsync_controller_t fsync_ctrl;
    fsync_config_t fsync_cfg;
    eu_controller_t eu_ctrl;
    eu_config_t eu_cfg;

    fsync_cfg.hartid = get_hartid();
    fsync_ctrl.base = NULL;
    fsync_ctrl.cfg = &fsync_cfg;
    fsync_ctrl.api = &fsync_api;

    eu_cfg.hartid = get_hartid();
    eu_ctrl.base = NULL;
    eu_ctrl.cfg = &eu_cfg;
    eu_ctrl.api = &eu_api;

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WFE);
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
