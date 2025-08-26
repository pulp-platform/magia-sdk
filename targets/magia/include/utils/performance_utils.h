#ifndef PERFORMANCE_UTILS_H
#define PERFORMANCE_UTILS_H


/**
 * @brief Starts all performance counters
 */
static inline void perf_start(void) {
    // enable all counters
    asm volatile("csrc 0x320, %0" : : "r"(0xffffffff));
    // arbitrary association of one event to one counter,
    // just the implemented ones will increase
    asm volatile("csrw 0x323, %0" : : "r"(1<<2));
}

/**
 * @brief Stops all performance counters
 */
static inline void perf_stop(void) {
    asm volatile("csrw 0x320, %0" : : "r"(0xffffffff));
}

/**
 * @brief Resets all performance counters to 0 without stopping them
 */
static inline void perf_reset(void) {
    perf_stop();
    asm volatile("csrw 0x320, %0" : : "r"(0));
}

/**
 * @brief Returns the cycles of the performance counter
 */
static inline unsigned int perf_get_cycles(){
    unsigned int value = 0;
    asm volatile ("csrr %0, 0xB00" : "=r" (value));
    return value;
}

/**
 * @brief Returns the n. instructions of the performance counter
 */
static inline unsigned int perf_get_instr(){
    unsigned int value = 0;
    asm volatile ("csrr %0, 0xB02" : "=r" (value));
    return value;
}


#endif