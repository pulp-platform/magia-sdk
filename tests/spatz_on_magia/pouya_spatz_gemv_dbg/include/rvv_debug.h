#ifndef RVV_DEBUG_H
#define RVV_DEBUG_H

#include <stdint.h>

#define RVV_DEBUG 1

#if RVV_DEBUG

static int asm_cnt = 0;

#define ASM_BEGIN(name)                                     \
do {                                                        \
    printf("\n");                                           \
    printf("=========================================================\n");\
    printf("[ASM %03d] %s\n", ++asm_cnt, name);             \
    printf("=========================================================\n");\
} while(0)

#define PRINT_SIZE(name,val) \
    printf("  %-12s = %zu\n", name, (size_t)(val))

#define PRINT_INT(name,val) \
    printf("  %-12s = %d\n", name, (int)(val))

#define PRINT_PTR(name,val) \
    printf("  %-12s = %p\n", name, (void *)(val))

#define PRINT_FP16(name,val) \
    printf("  %-12s = %8.5f\n", name, (float)(val))

#else

#define ASM_BEGIN(x)
#define PRINT_SIZE(a,b)
#define PRINT_INT(a,b)
#define PRINT_PTR(a,b)
#define PRINT_FP16(a,b)

#endif

#endif