#ifndef ONNX_CLIP_PARAMS_H_
#define ONNX_CLIP_PARAMS_H_

#include <stdint.h>
typedef struct {
    uintptr_t chunk_input;  /* Tile's input chunk                                       */
    uintptr_t chunk_res;    /* Tile's computed result chunk                             */
    uintptr_t chunk_exp;    /* Tile's golden model chunk                                */
    uintptr_t min;          /* Minimum value, under which element is replaced by min    */
    uintptr_t max;          /* Maximum value, above which element is replaced by max    */
    uint32_t start;         /* Tile's chunk global start index                          */
    uint32_t end;           /* Tile's chunk global end index                            */
    uint32_t len;           /* Tile's chunk len                                         */
} onnx_clip_params_t;

#endif  /* ONNX_CLIP_PARAMS_H_ */
