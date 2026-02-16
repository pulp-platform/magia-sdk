/*
 * Copyright (C) 2023-2025 ETH Zurich and University of Bologna
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
 * Authors: Alberto Dequino <alberto.dequino@unibo.it>
 * 
 * MAGIA Atomic Memory Operations Utils
 */

#ifndef AMO_UTILS_H
#define AMO_UTILS_H

#include "magia_tile_utils.h"

typedef struct node {
    struct node *next;
    int locked;
} lock_node;

/**
 * Atomically increase the value stored in addr by an immediate value
 */
int amo_add_immediate(uint32_t addr, uint32_t immediate){
    asm volatile("addi t0, %0, 0" ::"r"(addr));
    asm volatile("mv t1, %0" ::"r"(immediate));
    asm volatile("amoadd.w t2, t1, (t0)" ::);
    return 0;
}

/**
 * Atomically increase the value in addr by 1
 */
inline void amo_increment(volatile uint32_t addr){
    asm volatile("addi t0, %0, 0" ::"r"(addr));
    asm volatile("li t1, 1" ::);
    asm volatile("amoadd.w t2, t1, (t0)" ::);
}

/**
 * Atomic lock.
 * The lock is implemented as a linked list of lock_nodes + tail pointer.
 * The arguments are:
 * tail = the L1 pointer to the last lock_node that called this function on the lock.
 * mynode = the L1 pointer to the lock_node that wants to claim the lock.
 * First off, mynode will atomically swap tail's data (A) with his own pointer.
 * If the data received (A) is NULL, it means the lock's list was empty, therefore mynode now owns the lock and returns.
 * Otherwise, it means that the lock is in use.
 * In this case:
 * 1 - mynode will set his own "lock" to 1. 
 * 2 - mynode sets itself as the "next" node to (A). 
 * 3 - mynode will wait until his "lock" is set back to 0 by (A).
 * Once "lock" is set to 0, mynode owns the lock, and can return.
 */
void amo_lock(volatile uint32_t tail, volatile uint32_t mynode){
    ((volatile lock_node*)(mynode)) -> next = NULL;
    volatile uint32_t prev_node;
    asm volatile("addi t0, %0, 0" ::"r"(tail):"t0");
    asm volatile("addi t1, %0, 0" ::"r"(mynode):"t1");
    asm volatile("addi t2, %0, 0" ::"r"(mynode):"t2");
    asm volatile("amoswap.w t2, t1, (t0)" :::"t2", "t1", "t0");
    asm volatile("mv %0, t2" :"=r"(prev_node)::"t2");
    if(prev_node != NULL){
        ((volatile lock_node*)(mynode)) -> locked = 1;
        ((volatile lock_node*)(prev_node)) -> next = ((volatile lock_node*)(mynode));
        while(((volatile lock_node*)(mynode)) -> locked == 1) {};
    }
    return;
}

/**
 * Atomic unlock.
 * tail and mynode are the same as the previous call.
 * This function is paired to the previous lock function to free the lock owned by mynode.
 * First off, mynode will check if its "next" parameter has been changed by another lock_node.
 * If it is, then mynode has to set the "lock" parameter of the "next" lock_node to 0 to give him the lock.
 * Otherwise, it will try to atomically set the tail to NULL, to completely free the list.
 * In case this fails (because a new node has registered to the tail), then mynode will wait until the new node changes his "next" pointer.
 * Then, it will free the new lock_node, then return.
 */
void amo_unlock(volatile uint32_t tail, volatile uint32_t mynode){
    volatile uint32_t curr_tail;
    volatile uint32_t retval = 0;
    if(((volatile lock_node*)(mynode)) -> next == NULL){
        asm volatile("addi t1, %0, 0" ::"r"(tail):"t1");
        asm volatile("lr.w t0, (t1)" :::"t0", "t1");
        asm volatile("mv %0, t0" :"=r"(curr_tail)::"t0");
        if(curr_tail == mynode){
            asm volatile("sc.w t0, x0, (t1)" :::"t1", "t0");
            asm volatile("mv %0, t0" :"=r"(retval)::"t0");
            if(retval == 0)
                return;
            else{
                while(((volatile lock_node*)(mynode)) -> next == NULL);
            }
        }
    }
    ((volatile lock_node*)(mynode)) -> next -> locked = 0;
}

#endif //AMO_UTILS_H