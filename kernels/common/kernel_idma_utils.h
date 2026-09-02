#ifndef KERNEL_IDMA_UTILS_H_
#define KERNEL_IDMA_UTILS_H_

#include "eventunit.h"
#include "idma.h"

static inline void idma_ctrl_init(idma_controller_t *ctrl)
{
    ctrl->base = NULL;
    ctrl->cfg  = NULL;
    ctrl->api  = &idma_api;
}

static inline void eu_ctrl_init(eu_controller_t *ctrl)
{
    ctrl->base = NULL;
    ctrl->cfg  = NULL;
    ctrl->api  = &eu_api;
}

#endif /* KERNEL_IDMA_UTILS_H */
