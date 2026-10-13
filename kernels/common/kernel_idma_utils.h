#ifndef KERNEL_IDMA_UTILS_H_
#define KERNEL_IDMA_UTILS_H_

#include "eventunit.h"
#include "idma.h"

/*
 * The kernels' host-side sources call these before programming the iDMA or
 * waiting on the event unit. The HAL is controller-free, so there is no
 * controller state to set up any more; they are kept so the sources (and the
 * Deeploy templates that compile them) build unchanged.
 */
static inline void idma_ctrl_init(void) {}

static inline void eu_ctrl_init(void) {}

#endif /* KERNEL_IDMA_UTILS_H_ */
