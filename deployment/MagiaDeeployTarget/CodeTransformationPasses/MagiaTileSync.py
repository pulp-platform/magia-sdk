# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, _NoVerbosity

magiaSyncTilesFunctionTemplate = NodeTemplate("""
static void magia_sync_tiles_init(fsync_controller_t *fsync_ctrl, fsync_config_t *fsync_cfg,
                                  eu_controller_t *eu_ctrl, eu_config_t *eu_cfg) {
    fsync_cfg->hartid = get_hartid();
    fsync_ctrl->base = NULL;
    fsync_ctrl->cfg = fsync_cfg;
    fsync_ctrl->api = &fsync_api;

    eu_cfg->hartid = get_hartid();
    eu_ctrl->base = NULL;
    eu_ctrl->cfg = eu_cfg;
    eu_ctrl->api = &eu_api;
}

static inline void magia_sync_tiles(fsync_controller_t *fsync_ctrl, eu_controller_t *eu_ctrl) {
    fsync_sync_global(fsync_ctrl);
    eu_fsync_wait(eu_ctrl, WFE);
}
""")

magiaSyncTilesPrologueTemplate = NodeTemplate("""
fsync_controller_t magia_fsync_ctrl;
fsync_config_t magia_fsync_cfg;
eu_controller_t magia_eu_ctrl;
eu_config_t magia_eu_cfg;
magia_sync_tiles_init(&magia_fsync_ctrl, &magia_fsync_cfg, &magia_eu_ctrl, &magia_eu_cfg);
""")

_syncTilesTemplate = NodeTemplate("""
magia_sync_tiles(&magia_fsync_ctrl, &magia_eu_ctrl);
""")


class MagiaSynchTilesPass(CodeTransformationPass):

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        executionBlock.addRight(_syncTilesTemplate, {})
        return ctxt, executionBlock
