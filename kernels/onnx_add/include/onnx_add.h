#ifndef ONNX_ADD_H_
#define ONNX_ADD_H_

#include <stdint.h>

void MAGIA_onnx_add(const float16 *A, const float16 *B, float16 *C, uint32_t size);

#endif /* ONNX_ADD_H_ */
