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
        if((*(volatile _Float16*)(prev + (i * 2))) > (*(volatile _Float16*)(curr + (i * 2))))
            (*(volatile _Float16*)(curr + (i * 2))) = (*(volatile _Float16*)(prev + (i * 2)));
    }
}

/**
 * Finds the max value for each row and saves it in the result in the maxes buffer.
 */
int rowmax(uint32_t s, uint32_t maxes, uint32_t dim_h, uint32_t dim_w){
    for(uint32_t i = 0; i < dim_h; i++){
        uint32_t row = s + i * dim_w * 2;
        _Float16 rowmax = 0;
        for(uint32_t j = 0; j < dim_w; j++){
            if((*(volatile _Float16*)(row + j * 2)) > rowmax)
                rowmax = *(volatile _Float16*)(row + j * 2);
        }
        (*(volatile _Float16*)(maxes + i * 2)) = rowmax;
    }
}

/**
 * For each row i of the input h x w matrix "s", substract the i-th element of the "m" vector. 
 */
int rowdiff(uint32_t s, uint32_t m, uint32_t h, uint32_t w){
    for(uint32_t i = 0; i < h; i++){
        uint32_t row = s + i * w * 2;
        _Float16 diff = *(volatile _Float16*)(m + i * 2);
        for(uint32_t j = 0; j < w; j++){
            (*(volatile _Float16*)(row + j * 2)) = (*(volatile _Float16*)(row + j * 2)) - diff;
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
            sum = sum + *(volatile _Float16*)(row + j * 2);
        }
        (*(volatile _Float16*)(l + i * 2)) = sum;
    }
}

/**
 * For each row i of the input h x w matrix "s", divide the values by the i-th element of the "m" vector.
 */
int rowdiv(uint32_t s, uint32_t m, uint32_t h, uint32_t w){
    for(uint32_t i = 0; i < h; i++){
        uint32_t row = s + i * w * 2;
        _Float16 div = *(volatile _Float16*)(m + i * 2);
        for(uint32_t j = 0; j < w; j++){
            (*(volatile _Float16*)(row + j * 2)) = (*(volatile _Float16*)(row + j * 2)) / div;
        }
    }
}

/**
 * Element wise sum of v2 into v1
 */
int vect_sum(uint32_t v1, uint32_t v2, uint32_t dim){
    for(uint32_t i = 0; i < dim; i++){
        (*(volatile _Float16*)(v1 + i * 2)) = *(volatile _Float16*)(v1 + i * 2) + *(volatile _Float16*)(v2 + i * 2);
    }
}

/**
 * Element wise diff of v2 into v1
 */
int vect_diff(uint32_t v1, uint32_t v2, uint32_t dim){
    for(uint32_t i = 0; i < dim; i++){
        (*(volatile _Float16*)(v1 + i * 2)) = *(volatile _Float16*)(v1 + i * 2) - *(volatile _Float16*)(v2 + i * 2);
    }
}

/**
 * Element wise product of v2 into v1
 */
int vect_prod(uint32_t v1, uint32_t v2, uint32_t dim){
    for(uint32_t i = 0; i < dim; i++){
        (*(volatile _Float16*)(v1 + i * 2)) = (*(volatile _Float16*)(v1 + i * 2)) * (*(volatile _Float16*)(v2 + i * 2));
    }
}

#define GIST_A  12102203.17133801f
#define GIST_B  1064986823.010288f
#define GIST_C  8388608
#define GIST_D  2139095040

float fastexp_gist(float x) {
    x = GIST_A * x + GIST_B;

    if (x < GIST_C || x > GIST_D)
        x = (x < GIST_C) ? 0.0f : GIST_D;

    uint32_t n = (uint32_t)(x);
    return *(float *) &n;
}

int exponential(uint32_t matrix, uint32_t rows, uint32_t columns){
    for(uint32_t i = 0; i < rows; i++){
        for(uint32_t j = 0; j < columns; j++){
            (*(volatile _Float16 *)(matrix + i * columns * 2 + j * 2)) = (_Float16) fastexp_gist((float) (*(volatile _Float16 *)(matrix + i * columns * 2 + j * 2)));
        }
    }
}


#endif //ATTENTION_UTILS_H