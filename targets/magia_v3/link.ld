/*
 * Copyright (C) 2018-2019 ETH Zurich and University of Bologna
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
 */

SEARCH_DIR(.)
__DYNAMIC  =  0;

/* L2 spans 0xc0000000 .. 0xccfeffff (see gvsoc magia_v3 arch.py L2_ADDR_START/L2_SIZE).
 * The program image sits at the top of it: dataram ends just below TEST_END_ADDR
 * (0xccff0000) and so cannot grow. l2_bulk is the 192 MB of L2 *below* the image,
 * mapped both from the tile's OBI xbar and from the NoC (so the iDMA reaches it), for
 * payloads too large to live in dataram - e.g. a whole network's weights. Tests that
 * place nothing there emit no such section and are unaffected. */
MEMORY
{
    l2_bulk     : ORIGIN = 0xc0000000, LENGTH = 0x0C000000
    /* instrram holds .text plus the embedded Spatz and PULP binaries. It used to stop at
     * 0x8000, leaving the 32 KB up to dataram unused; a test that links several kernels
     * into one Spatz binary does not fit in that. onnx_resnet18 (seven kernels) needed
     * 0x10000, which onnx_mobilevit_v2 (eleven kernels, twelve Spatz tasks) then filled
     * to 96%, so it is 0x20000 now. Growing it moves dataram up rather than eating into
     * it: dataram uses well under 1% of its 15.7 MB in every test here, and its end stays
     * below TEST_END_ADDR (0xccff0000) with 832 KB to spare. */
    instrram    : ORIGIN = 0xcc000000, LENGTH = 0x20000
    dataram     : ORIGIN = 0xcc020000, LENGTH = 0xF00000
    stack       : ORIGIN = 0x00010000, LENGTH = 0xC000 /* 16K for control core CV32 */
}

/* Stack information variables */
_min_stack     = 0x1000;   /* 4K - minimum stack space to reserve */
_stack_len     = LENGTH(stack);
_stack_start   = ORIGIN(stack) + LENGTH(stack);

/* We have to align each sector to word boundaries as our current s19->slm
 * conversion scripts are not able to handle non-word aligned sections. */

PHDRS
{
    bulk PT_LOAD;
    text PT_LOAD;
    data PT_LOAD;
}

SECTIONS
{
    /* Read-only bulk payload, loaded from the ELF. */
    .l2_bulk : {
        . = ALIGN(4);
        _l2_bulk_start = .;
        KEEP(*(.l2_bulk))
        KEEP(*(.l2_bulk.*))
        _l2_bulk_end = .;
    } > l2_bulk :bulk

    /* MAPS application state must remain in the directly addressed application-data
     * window.  Consume its named subsection before the generic network arena below. */
    .l2_arena (NOLOAD) : {
        . = ALIGN(4);
        _l2_arena_start = .;
        *(.l2_arena.maps_application)
        _l2_arena_end = .;
    } > dataram :data

    /* Large model scratch payload, NOLOAD so it never enters the ELF. Nothing zeroes
     * it, so anything placed here must be written before it is read. */
    .l2_model_arena (NOLOAD) : {
        . = ALIGN(4);
        _l2_model_arena_start = .;
        *(.l2_arena)
        *(.l2_arena.*)
        _l2_model_arena_end = .;
    } > l2_bulk

    .vectors :
    {
        . = ALIGN(4);
        KEEP(*(.vectors))
    } > instrram :text

    .text : {
        . = ALIGN(4);
        _stext = .;
        *(.text)
        _etext  =  .;
        __CTOR_LIST__ = .;
        LONG((__CTOR_END__ - __CTOR_LIST__) / 4 - 2)
        *(.ctors)
        LONG(0)
        __CTOR_END__ = .;
        __DTOR_LIST__ = .;
        LONG((__DTOR_END__ - __DTOR_LIST__) / 4 - 2)
        *(.dtors)
        LONG(0)
        __DTOR_END__ = .;
        *(.lit)
        *(.shdata)
        _endtext = .;
    }  > instrram :text
	
	/* Spatz embedded binary - inlined from header file */
    /* Positioned right after CV32 .text in instrram */

    . = ALIGN(4);
    _spatz_binary_start = .;

    .spatz_binary : {
        KEEP(*(.spatz_binary))
    } > instrram :text

    _spatz_binary_end = .;
    _spatz_binary_size = _spatz_binary_end - _spatz_binary_start;

    /* PULP embedded binary - inlined from header file */
    /* Positioned right after Spatz binary in instrram */

    . = ALIGN(4);
    _pulp_binary_start = .;

    .pulp_binary : {
        KEEP(*(.pulp_binary))
    } > instrram :text

    _pulp_binary_end = .;
    _pulp_binary_size = _pulp_binary_end - _pulp_binary_start;

    /*--------------------------------------------------------------------*/
    /* Global constructor/destructor segment                              */
    /*--------------------------------------------------------------------*/

    .preinit_array     :
    {
      PROVIDE_HIDDEN (__preinit_array_start = .);
      KEEP (*(.preinit_array))
      PROVIDE_HIDDEN (__preinit_array_end = .);
    } > dataram :data

    .init_array     :
    {
      PROVIDE_HIDDEN (__init_array_start = .);
      KEEP (*(SORT(.init_array.*)))
      KEEP (*(.init_array ))
      PROVIDE_HIDDEN (__init_array_end = .);
    } > dataram :data

    .fini_array     :
    {
      PROVIDE_HIDDEN (__fini_array_start = .);
      KEEP (*(SORT(.fini_array.*)))
      KEEP (*(.fini_array ))
      PROVIDE_HIDDEN (__fini_array_end = .);
    } > dataram :data

    .data : {
        . = ALIGN(4);
        sdata  =  .;
        _sdata  =  .;
        *(.data);
        *(.data.*)
        edata  =  .;
        _edata  =  .;
    } > dataram :data

    .rodata : {
        . = ALIGN(4);
        *(.rodata);
        *(.rodata.*)
    } > dataram :data

    .shbss :
    {
        . = ALIGN(4);
        *(.shbss)
    } > dataram :data

    .bss :
    {
        . = ALIGN(4);
        _bss_start = .;
        *(.bss)
        *(.bss.*)
        *(.sbss)
        *(.sbss.*)
        *(COMMON)
        _bss_end = .;
    } > dataram :data

    /* Reserve the complete control-core stack window.  crt0 initializes sp from
     * `stack`, so this symbol must denote the top of the downward-growing stack. */
    .stack (NOLOAD): {
        . = ALIGN(4);
        . = . + _stack_len;
        . = ALIGN(4);
        stack = . ;
        _stack = . ;
    } > stack :NONE

}
