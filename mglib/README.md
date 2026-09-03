This folder contains the Magia Library **(mglib)**, which acts as a wrapper for programming the [MAGIA](https://github.com/pulp-platform/MAGIA/tree/main) mesh using the [MAGIA-SDK](https://github.com/pulp-platform/magia-sdk) APIs and HALs in an efficient and ordered manner.

## Features

Currently the **mglib** support programming for the **IDMA** and **RedMule** hardware accelerators available on the [MAGIA](https://github.com/pulp-platform/MAGIA/tree/main) platform.

Each operation issued by this library is identified and managed through a Magia Event **(mg_event)** object containing:
- An ID to identify the event
- A Callback Handler which is invoked when the event is fired

Based on the specific accelerator, the initialization and usage of these software events is different and explained below.

Events are managed and checked through a **Event Unit**, to enable it you have to include:

- The MAGIA tile runtime library 

`#include "tile.h"`

- The MAGIA Event unit library

`#include "mg_event.h"`

The event unit is managed by a controller, you can instantiate it like this:

```
eu_config_t eu_cfg      = {.hartid = hartid};
eu_controller_t eu_ctrl = {
    .base = NULL,
    .cfg  = &eu_cfg,
    .api  = &eu_api,
};
eu_init(&eu_ctrl);
```

It is necessary to instantiate a number **mg_event_t** objects before calling the **mglib** functions. It is possible to re-use a finished event to program a different operation, but you need multiple instances in order to enqueue or parallelize them.

# REDMULE Programming model

**RedMule** allows enqueuing up to 2 different jobs at the same time to be executed.

To enable this feature, it is necessary to first include the MAGIA RedMule library

`#include "mg_redmule.h"`

Then instantiate and initialize the Redmule controller like this:

```
redmule_config_t redmule_cfg      = {.hartid = hartid};
redmule_controller_t redmule_ctrl = {
    .base = NULL,
    .cfg  = &redmule_cfg,
    .api  = &redmule_api,
};

redmule_init(&redmule_ctrl);
```

Finally, you need to enable the RedMule events on the Event Unit controller instantiated earlier:

`eu_redmule_init(&eu_ctrl, 0);`

Once you have both the controllers ready and instantiated a **mg_event_t**, you can start programming and enqueing GEMM operations (MxN * NxK) like this:

```
mg_redmule_gemm_enqueue(&redmule_ctrl,
                        &eu_ctrl,
                        WAIT_MODE,
                        &input_matrix_0,
                        &input_matrix_1,
                        &output_matrix,
                        M_SIZE,
                        N_SIZE,
                        K_SIZE,
                        &redmule_evt,
                        redmule_callback_t);
```

Doing so enqueues the GEMM in RedMule's FIFO, but does not free the controller for pushing another task. After preparing the task, it is necessary to commit it with one of the following calls:

```
mg_redmule_gemm_commit(&redmule_ctrl);

mg_redmule_gemm_commit_start(&redmule_ctrl);
```

The first call does NOT start the task, whereas the second triggers it.
To start the job (if there is one in the queue):

`mg_redmule_gemm_start(&redmule_ctrl);`

It is then possible to wait for a specific software event by passing the corresponding mg_even_t as a parameter:

`mg_redmule_wait(&eu_ctrl, WAIT_MODE, redmule_evt_curr);`

Supported WAIT_MODEs are WFE and POLLING.

# IDMA Programming model

Unlike RedMule, the IDMA doesn't support in hardware enqueueing multiple transfers on the same channel. Therefore, a workaround SW-based 1-depth queue has been implemented.

It is necessary to include the Magia IDMA library:

`#include "mg_idma.h"`

And instantiate the IDMA controller:

```
idma_config_t idma_cfg      = {.hartid = hartid};
idma_controller_t idma_ctrl = {
    .base = NULL,
    .cfg  = &idma_cfg,
    .api  = &idma_api,
};
idma_init(&idma_ctrl);
```

Then also enable IDMA events on the Event Unit:

`eu_idma_init(&eu_ctrl, 0);`

Once this is done, it is possible to program and run 2D IDMA memory transfers via the following function call:

```
mg_idma_memcpy_2d(&idma_ctrl,
                  &eu_ctrl,
                  WAIT_MODE,
                  dir,
                  axi_addr,
                  obi_addr,
                  len,
                  std,
                  reps,
                  &idma_evt,
                  &callback);
```

Where dir=0 means using the AXI -> OBI channel (read), and dir=1 means OBI -> AXI channel (write).
Other parameters:
- len: dimension of each contigous block of the 2D transfer
- std: len + gap size between each block (stride)
- reps: number of blocks to be transfered

The IDMA call is run ASAP in a non-blocking fashion. To verify it's execution, you can wait on its corresponding event:

`mg_idma_wait(&eu_ctrl, 0, WAIT_MODE, &idma_evt);`