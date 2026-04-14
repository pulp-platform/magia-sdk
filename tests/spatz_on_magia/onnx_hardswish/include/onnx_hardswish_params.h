#ifndef ONNX_HARDSWISH_PARAMS_H_
#define ONNX_HARDSWISH_PARAMS_H_

#include <stdint.h>

typedef struct {
    uintptr_t chunk_X;      /* Tile's input chunk               */
    uintptr_t chunk_Y;      /* Tile's computed result chunk     */
    uintptr_t chunk_G;      /* Tile's golden model chunk        */
    uint32_t start;         /* Tile's chunk global start index  */
    uint32_t end;           /* Tile's chunk global end index    */
    uint32_t len;           /* Tile's chunk len                 */
} onnx_hardswish_params_t;

#endif  /* ONNX_HARDSWISH_PARAMS_H_ */
