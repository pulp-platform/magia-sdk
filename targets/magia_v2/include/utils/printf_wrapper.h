#ifndef PRINTF_WRAPPER_H_
#define PRINTF_WRAPPER_H_

// Spatz tasks are built standalone (-nostdlib), use tinyprintf there.
#ifdef SPATZ_TARGET
#include "tinyprintf.h"
#else
#include "printf.h"
#endif

#endif  /* PRINTF_WRAPPER_H_ */
