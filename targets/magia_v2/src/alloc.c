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

#include <stdint.h>
#include <stddef.h>

#include "utils/alloc.h"
#include "addr_map/tile_addr_map.h"

extern void *_sl2_heap;
extern void *_el2_heap;

static uint8_t *l2_heap_end = (uint8_t *)&_el2_heap;
static uint8_t *l2_heap_start = (uint8_t *)&_sl2_heap;

static MemoryBlock *l2_heap_freelist = NULL;

/**
 * THIS MALLOC IS NOT THREAD SAFE!
 * 
 * Make sure only one tile is only accessing this malloc at a time, either by using amo locks/barriers,
 * or by synchronizing.
 */
static void *region_malloc(uint8_t **ptr, uint8_t *end, MemoryBlock **freelist, size_t size) {
    size_t total_size = ALLOC_ALIGN(size + sizeof(MemoryBlock));

    MemoryBlock **prev = freelist;
    MemoryBlock *curr = *freelist;

    while (curr) {
        if (curr->size >= size) {
            *prev = curr->next;
            return (void *)(curr + 1);
        }
        prev = &curr->next;
        curr = curr->next;
    }

    if (*ptr + total_size > end) return NULL;

    MemoryBlock *b = (MemoryBlock *)(*ptr);
    b->size = size;
    *ptr += total_size;

    return (void *)(b + 1);
}

static void region_free(MemoryBlock **freelist, void *ptr) {
    if (!ptr) return;
    MemoryBlock *b = (MemoryBlock *)((uint8_t *)ptr - sizeof(MemoryBlock));
    b->next = *freelist;
    *freelist = b;
}

void *magia_l2_malloc(size_t size) {
    return region_malloc(&l2_heap_start, l2_heap_end, &l2_heap_freelist, size);
}

void magia_l2_free(void *ptr){
    region_free(&l2_heap_freelist, ptr);
}