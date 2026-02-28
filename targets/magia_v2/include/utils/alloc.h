/*
 * Copyright (C) 2023-2026 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Authors:
 * Alberto Dequino <alberto.dequino@unibo.it>
 * 
 * MAGIA allocator
 */

#ifndef ALLOC_H
#define ALLOC_H


#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/*
 * All allocated memory blocks are aligned to this boundary for optimal access performance.
 */
#define ALLOC_ALIGNMENT 4

/**
 * Macro to align a size to the memory alignment boundary.
 */
#define ALLOC_ALIGN(size) (((size) + (ALLOC_ALIGNMENT - 1)) & ~(ALLOC_ALIGNMENT - 1))

/**
 * Memory block structure for the freelist allocator.
 *
 * This structure is used internally to track free memory blocks in a linked list.
 * Each allocated block has this header prepended to store metadata.
 */
typedef struct MemoryBlock {
    struct MemoryBlock *next; /**< Pointer to the next free block in the list. */
    size_t size;              /**< Size of the memory block in bytes. */
} MemoryBlock;

/**
 * This function allocates a block of memory of the specified size from the dataram heap.
 * The allocated memory is aligned to ALLOC_ALIGNMENT bytes. The memory is not initialized.
 *
 * The allocated memory should be freed using magia_l2_free() when no longer needed.
 */
void *magia_l2_malloc(size_t size);

/**
 * This function returns a memory block to the free list, making it available for future
 * allocations. The memory block is added to the head of the freelist for efficient reuse.
 *
 * The pointer must have been returned by a previous call to magia_l2_malloc().
 * After calling this function, the memory pointed to by ptr should not be accessed.
 */
void magia_l2_free(void *ptr);

/**
 * This function allocates memory for an array of num elements, each of size bytes,
 * and initializes all bytes to zero.
 *
 * The allocated memory should be freed using magia_l2_free() when no longer needed.
 */
void *magia_l2_calloc(size_t num, size_t size);

/**
 * This function changes the size of the memory block pointed to by ptr to size bytes.
 * The contents are unchanged up to the minimum of the old and new sizes.
 * If the new size is larger, the additional memory is uninitialized.
 *
 * If reallocation fails, the original memory block is unchanged.
 */
void *magia_l2_realloc(void *ptr, size_t size);

/**
 * Gets the total size of the dataram l2 heap.
 */
size_t magia_l2_heap_size(void);

/**
 * This function calculates the total amount of free memory by traversing the freelist
 * and summing the sizes of all free blocks, plus any remaining unallocated heap space.
 */
size_t magia_l2_heap_free(void);

/**
 * Gets the amount of allocated memory in the l2 heap.
 */
size_t magia_l2_heap_used(void);

/**
 * This function verifies if the given pointer points to a memory block that was
 * allocated by magia_l2_malloc() and is within the heap bounds.
 */
bool magia_l2_ptr_valid(void *ptr);

/**
 * This function validates the integrity of the heap data structures, including:
 * - Freelist consistency
 * - Memory block header validity
 * - Heap boundary checks
 *
 * This function is intended for debugging and should not be used in production code
 * due to its performance impact.
 */
bool magia_l2_heap_check(void);


#endif //ALLOC_H






