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
 * MAGIA Attention ISA Utils
 */

#ifndef ATTENTION_UTILS_H
#define ATTENTION_UTILS_H

#include "magia_tile_utils.h"

 /**
 * Element-wise comparison of the max vectors.
 * Saves in the curr buffer the bigger values.
 */
int max_compare(uint32_t curr, uint32_t prev, uint32_t dim){
    for(uint32_t i = 0; i < dim; i++){
        if((*(volatile uint16_t*)(prev + (i * 2))) > (*(volatile uint16_t*)(curr + (i * 2))))
            mmio16(curr + (i * 2)) = (*(volatile uint16_t*)(prev + (i * 2)));
    }
}

/**
 * Finds the max value for each row and saves it in the result in the maxes buffer.
 */
int rowmax(uint32_t s, uint32_t maxes, uint32_t dim_h, uint32_t dim_w){
    for(uint32_t i = 0; i < dim_h; i++){
        uint32_t row = s + i * dim_w * 2;
        uint16_t rowmax = 0;
        for(uint32_t j = 0; j < dim_w; j++){
            if((*(volatile uint16_t*)(row + j * 2)) > rowmax)
                rowmax = *(volatile uint16_t*)(row + j * 2);
        }
        mmio16(maxes + i * 2) = rowmax;
    }
}

/**
 * For each row i of the input h x w matrix "s", substract the i-th element of the "m" vector. 
 */
int rowdiff(uint32_t s, uint32_t m, uint32_t h, uint32_t w){
    for(uint32_t i = 0; i < h; i++){
        uint32_t row = s + i * w * 2;
        uint16_t diff = *(volatile uint16_t*)(m + i * 2);
        for(uint32_t j = 0; j < w; j++){
            mmio16(row + j * 2) = (*(volatile uint16_t*)(row + j * 2)) - diff;
        }
    }
}

/**
 * For each row i of the input h x w matrix "s", sum the values and store it in the i-th element of the "l" vector.
 */
int rowsum(uint32_t s, uint32_t l, uint32_t h, uint32_t w){
    for(uint32_t i = 0; i < h; i++){
        uint32_t row = s + i * 2 * w;
        uint16_t sum = 0;
        for(uint32_t j = 0; j < w; j++){
            sum = sum + *(volatile uint16_t*)(row + j * 2);
        }
        mmio16(l + i * 2) = sum;
    }
}

/**
 * For each row i of the input h x w matrix "s", divide the values by the i-th element of the "m" vector.
 */
int rowdiv(uint32_t s, uint32_t m, uint32_t h, uint32_t w){
    for(uint32_t i = 0; i < h; i++){
        uint32_t row = s + i * w * 2;
        uint16_t div = *(volatile uint16_t*)(m + i * 2);
        for(uint32_t j = 0; j < w; j++){
            mmio16(row + j * 2) = (*(volatile uint16_t*)(row + j * 2)) / div;
        }
    }
}

/**
 * Element wise sum of v2 into v1
 */
int vect_sum(uint32_t v1, uint32_t v2, uint32_t dim){
    for(uint32_t i = 0; i < dim; i++){
        mmio16(v1 + i * 2) = *(volatile uint16_t*)(v1 + i * 2) + *(volatile uint16_t*)(v2 + i * 2);
    }
}

/**
 * Element wise diff of v2 into v1
 */
int vect_diff(uint32_t v1, uint32_t v2, uint32_t dim){
    for(uint32_t i = 0; i < dim; i++){
        mmio16(v1 + i * 2) = *(volatile uint16_t*)(v1 + i * 2) - *(volatile uint16_t*)(v2 + i * 2);
    }
}

/**
 * Element wise product of v2 into v1
 */
int vect_prod(uint32_t v1, uint32_t v2, uint32_t dim){
    for(uint32_t i = 0; i < dim; i++){
        mmio16(v1 + i * 2) = (*(volatile uint16_t*)(v1 + i * 2)) * (*(volatile uint16_t*)(v2 + i * 2));
    }
}


#endif //ATTENTION_UTILS_H