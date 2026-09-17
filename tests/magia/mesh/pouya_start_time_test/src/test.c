/*

    Using this code is free by citing the name author:
    Pouya Shirinshahrakfard
    Email: pouyashirinfard@gmail.com
    
    for more info, please contact the author.
*/   


#include <stdint.h>

#include "tile.h"
#include "idma.h"
#include "fsync.h"
#include "eventunit.h"
#include "test.h"

#define WAIT_MODE WFE


static uint32_t task_end[128] __attribute__((section(".l2"), aligned(64))) = {0};
static uint32_t L1_address[128] __attribute__((section(".l2"), aligned(64))) = {0};

int p = 0;

void loop (int i){

    for(int z=0; z<i; z++){
        for(int k=0; k<i; k++){
            for (int j=0; j<i; j++){
                if(j==k | j==z)        
                    p++;

            }
        }
    }
}

/*
--------------------------------------------------
L2 buffers for compressed data
(shared across cores)
--------------------------------------------------
*/
int main(void)
{
    uint32_t hartid = get_hartid();
    uint32_t y_id = GET_Y_ID(hartid);
    uint32_t x_id = GET_X_ID(hartid);

    /*
    --------------------------------------------------
    Init FSYNC
    --------------------------------------------------
    */
    fsync_config_t fsync_cfg = {
        .hartid = hartid
    };

    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg  = &fsync_cfg,
        .api  = &fsync_api,
    };

    fsync_init(&fsync_ctrl);

    /*
    --------------------------------------------------
    Init Event Unit
    --------------------------------------------------
    */
    eu_config_t eu_cfg = {
        .hartid = hartid
    };

    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg  = &eu_cfg,
        .api  = &eu_api,
    };

    eu_init(&eu_ctrl);
    eu_clear_events(0xFFFFFFFF);
    eu_fsync_init(&eu_ctrl, 0);

    int start = perf_get_cycles();
    perf_start();

    //loop(30);  //workload -> print null

    uint32_t l1 = get_l1_base(hartid);

    //barrier
    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);
    
    int end = perf_get_cycles();

    task_end[hartid] = end - start;
    L1_address[hartid] = l1;

    /*
    --------------------------------------------------
    Verify
    --------------------------------------------------
    */
    if (hartid == 0) {

        for(int i=0; i<64; i++)
            //printf("%u\n", task_end[i]);
            printf("Core[%u]  L1 address -> 0x%x\n", i, L1_address[i]);

        //printf("%u",p);


    }

    return 0;
}