/*
 * Copyright (C) 2023-2025 ETH Zurich and University of Bologna
 * Copyright (c) 1997-2010, 2012-2015 Wind River Systems, Inc.
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
 * PRINTF
 */
#ifndef PRINTF_H
#define PRINTF_H

#include <stdarg.h>
#include "prf.h"
#include "io.h"

static int printf(const char *format, ...)
{
	va_list vargs;
	int     r;

	va_start(vargs, format);
	r = prf(pputc, format, vargs);
	va_end(vargs);

	return r;
}

//#define printf tfp_printf

#endif // PRINTF_H
