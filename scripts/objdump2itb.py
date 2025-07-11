#!/usr/bin/env python3

################################################################################
#
# Copyright 2021 OpenHW Group
# Copyright 2021 Silicon Labs
# 
# Licensed under the Solderpad Hardware Licence, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://solderpad.org/licenses/
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier:Apache-2.0 WITH SHL-2.0
# 
################################################################################
#
# dump2itb: python script to parse a GNU binutils objdump output file
#           into an instruction table format, which is a custom
#           text file more suitable for parsing using the standard IEEE 1800 
#           text parsing functions
#
# The objdump should set the following flags to ensure that the regexps will 
# parse the expected text properly
# % objdump -d -l -s program.elf > program.objdump
# 
# The instruction table file format for a single instruction is:
# 
# #<num_of_src_lines>
# #<stc_line_0>
# #<stc_line_1>
# #...
# #<src_line_n>
# <instr_addr> <instr_func> <basename of instr_src> <line_num> <opcode> <num of asm words> <asm_word_0> <asm_word_1> <...> <asm_word_n> 
#
# An esample: Note that the pound signs are part of the format
# #6
# #       // Expect DCSR
# #       //   31:28 XDEBUGER Version = 4
# #       //    8:6   Cause           = 2 Trigger
# #       //    1:0   Privelege       = 3 Machine
# #       // TBD FIXME BUG documentation update needed
# #       li   t1, 4<<28 | 2<<6 | 3<<0 | 1<<15
# 437324064 _debugger_trigger_match_ebreak debugger.S 171 40008337 2 lui x6,0x40008
#
#
# Author: Steve Richmond
#  email: steve.richmond@silabs.com
#
# Written with Python 3.5.1 on RHEL 7.7.  Your python mileage may vary.
#
################################################################################

import string
import sys
import re
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CFunction:
    def __init__(self, addr, name):
        self.addr = addr
        self.name = name

# Regular expression to extract a function from objdump
# Example:
# 00000256 <end_handler_incr_mepc>:
FUNC_PATTERN     = "(?P<addr>[0-9a-f]{8}) <(?P<name>\S*)>:"
FUNC_RE          = re.compile(FUNC_PATTERN)

# Regular expression to extract an individual instruction
# Example:
#      264:       00a31363                bne     t1,a0,26a <end_handler_incr_mepc2>
INST_PATTERN     = "(?P<addr>[0-9a-f]{1,8}):\t*(?P<mcode>[0-9a-f]{4}([0-9a-f]{4})?)\s{2,}(?P<asm>[a-z].*)$"
INST_RE          = re.compile(INST_PATTERN)

# Regular expression to extract a source annotation for each instruction
# Example:
# /work/strichmo/core-v-verif/cv32e40x/tests/programs/custom/debug_test_trigger/debugger.S:47
SRC_FILE_PATTERN = "^(?P<dir>/\S+)/(?P<file>[^/\s]+):(?P<line>[0-9]*)$" 
SRC_FILE_RE      = re.compile(SRC_FILE_PATTERN)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('objdump', help='Required objdump file for parsing.  Use - to read STDIN')
parser.add_argument('--debug', action='store_true', help='Enable more debug messages')

# Main function
def objdump2itb():
    '''Parse an objdump file'''
    args = parser.parse_args(sys.argv[1:])
    if args.debug:
        logger.setLevel(logging.DEBUG)

    parse(args.objdump)

def parse(objdump):
    '''Parse the file'''

    if objdump == '-':
        fp = sys.stdin
    else:
        fp = open(objdump)

    current_cfunction = None
    current_src_file = None
    current_src_line = 0
    current_src_code = ""    
    used_src = 0

    for line in fp:
        line_orig = line.strip('\n')
        line = line.strip()
        if not line:
            continue

        # Parse function
        func = FUNC_RE.match(line)
        if func:
            d = func.groupdict()
            current_cfunction = CFunction(d["addr"], d["name"])
            current_src_code = ""   # reset the source code lines
            used_src = 0
            continue

        # Parse source file
        src_file = SRC_FILE_RE.match(line)
        if src_file:
            d = src_file.groupdict()
            current_src_file = d["file"]
            current_src_line = d["line"]  
            current_src_code = ""   # reset the source code lines
            used_src = 0
            continue

        # Parse instruction
        inst = INST_RE.match(line)
        if inst:
            d = inst.groupdict()
            addr = d["addr"]
            mcode = d["mcode"]
            asm = d["asm"]
            mcode = mcode.replace(" ", "")
            asm = asm.replace("\t", " ").strip()            
            addr = int("0x{}".format(addr), 0)
            fname = current_cfunction.name
            asm_len = len(asm.split())
            
            used_src = 1
            if current_src_code == "":
                src_len = 0
                print("#{}".format(src_len))
            else:
                src_len = len(current_src_code.split('\n')); 
                print("#{}\n{}".format(src_len, current_src_code))

            print("{} {} {} {} {} {} {}".format(addr, fname, current_src_file, current_src_line, mcode, asm_len, asm))
            
            logger.debug("Addr: '{:x}' Func: '{}' Source File: '{}' Source lineno: '{}' mcode: '{}' Asm: '{}'".format(
                           int(addr), fname, current_src_file, current_src_line, mcode, asm))
            continue

        # no match on filters
        if current_src_code:
            src_len = len(current_src_code.split('\n')); 
            if (src_len < 10):   # keep src from getting too long, also good if filenames are missing
                current_src_code += "\n#" + line_orig
        else:
            if current_src_file:    # don't start adding source before first src file
                # first src line no newline before
                current_src_code = "#" + line_orig
        
        logger.debug("Unmatched (treating as source line): '{}'".format(line))


if __name__ == '__main__':
    objdump2itb()