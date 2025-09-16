# Copyright (C) 2025 ETH Zurich and University of Bologna
#
# Licensed under the Solderpad Hardware License, Version 0.51 
# (the "License"); you may not use this file except in compliance 
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: SHL-0.51
#
# Authors: Victor Isachi <victor.isachi@unibo.it>
# Alberto Dequino <alberto.dequino@unibo.it>
# 
# Magia-sdk Makefile

SHELL 			:= /bin/bash

BUILD_DIR 		?= ../work/sw/tests/$(test).c
MAGIA_DIR 		?= ../
GVSOC_DIR 		?= ./gvsoc
BIN 			?= $(BUILD_DIR)/verif
build_mode		?= profile
fsync_mode		?= stall
mesh_dv			?= 1

target_platform ?= magia
compiler 		?= GCC_MULTILIB
gui 			?= 0
tiles 			?= 2

tiles_2 		:= $(shell echo $$(( $(tiles) * $(tiles) )))
tiles_log    	:= $(shell awk 'BEGIN { printf "%.0f", log($(tiles_2))/log(2) }')
tiles_log_real  := $(shell awk 'BEGIN { printf "%.0f", log($(tiles))/log(2) }')

.PHONY: gvsoc

clean:
	rm -rf build/

build:
	sed -i -E 's/^#define MESH_([XY])_TILES[[:space:]]*[0-9]+/#define MESH_\1_TILES $(tiles)/' ./targets/magia/include/addr_map/tile_addr_map.h
	sed -i -E 's/^(#define MAX_SYNC_LVL[[:space:]]*)[0-9]+/\1$(tiles_log)/' ./targets/magia/include/addr_map/tile_addr_map.h
	sed -i -E 's/^(#define MESH_2_POWER[[:space:]]*)[0-9]+/\1$(tiles_log_real)/' ./targets/magia/include/addr_map/tile_addr_map.h
ifeq ($(compiler), LLVM)
	$(error COMING SOON!)
endif
ifeq ($(compiler), GCC_PULP)
	sed -i -E 's/^add_subdirectory\(flatatt\)/#&/' ./tests/magia/mesh/CMakeLists.txt
	sed -i -E 's/^#include "utils\/attention_utils.h"/\/\/&/' ./targets/magia/include/tile.h
endif
ifeq ($(compiler), GCC_MULTILIB)
	sed -i -E 's/^#add_subdirectory\(flatatt\)/add_subdirectory\(flatatt\)/' ./tests/magia/mesh/CMakeLists.txt
	sed -i -E 's/^\/\/#include "utils\/attention_utils.h"/#include "utils\/attention_utils.h"/' ./targets/magia/include/tile.h
endif
	cmake -DTARGET_PLATFORM=$(target_platform) -DCOMPILER=$(compiler) -B build --trace-expand
	cmake --build build --verbose

set_mesh:
ifeq ($(tiles), 1)
	$(eval mesh_dv=0)
endif 

run: set_mesh
	@echo 'Magia is available at https://github.com/pulp-platform/MAGIA.git'
	@echo 'please run "source setup_env.sh" in the magia folder before running this script'
	@echo 'and make sure the risc-v objdump binary is visible on path using "which riscv32-unknown-elf-objdump".'
ifndef test
	$(error Proper formatting is: make run test=<test_name> platform=rtl|gvsoc)
endif
ifeq (,$(wildcard ./build/bin/$(test)))
	$(error No test found with name: $(test))
endif
ifndef platform
	$(error Proper formatting is: make run test=<test_name> platform=rtl|gvsoc)
endif
ifeq ($(platform), gvsoc)
	$(GVSOC_DIR)/install/bin/gvsoc --target=magia-base --binary=./build/bin/$(test) --trace-level=trace run
else ifeq ($(platform), rtl)
	mkdir -p $(BUILD_DIR)
	cp ./build/bin/$(test) $(BUILD_DIR)/verif
	objcopy --srec-len 1 --output-target=srec $(BIN) $(BIN).s19
	scripts/parse_s19.pl $(BIN).s19 > $(BIN).txt
	python3 scripts/s19tomem.py $(BIN).txt $(BUILD_DIR)/stim_instr.txt $(BUILD_DIR)/stim_data.txt
	cd $(BUILD_DIR)													&& \
	cp -sf ../../../../modelsim.ini modelsim.ini    				&& \
	ln -sfn ../../../../work work         			
	riscv32-unknown-elf-objdump -d -S $(BIN) > $(BIN).dump
	riscv32-unknown-elf-objdump -d -l -s $(BIN) > $(BIN).objdump
	python3 scripts/objdump2itb.py $(BIN).objdump > $(BIN).itb
	cd $(MAGIA_DIR) 												&& \
	make run test=$(test) gui=$(gui) mesh_dv=$(mesh_dv)
else
	$(error Only rtl and gvsoc are supported as platforms.)
endif

MAGIA: set_mesh
ifeq ($(target_platform), magia)
	sed -i -E 's/^(num_cores[[:space:]]*\?=[[:space:]]*)[0-9]+/\1$(tiles_2)/' $(MAGIA_DIR)/Makefile
ifneq ($(tiles), 1)
	sed -i -E 's/^( *localparam int unsigned N_TILES_[XY][[:space:]]*=[[:space:]]*)[0-9]+;/\1$(tiles);/' $(MAGIA_DIR)/hw/mesh/magia_pkg.sv
endif
else
	$(error unrecognized platform (acceptable platform: magia).)
endif
ifeq ($(fsync_mode), stall)
	sed -i -E "s/^(parameter bit[[:space:]]+FSYNC_STALL[[:space:]]+= )[01];/\11;/" $(MAGIA_DIR)/hw/tile/magia_tile_pkg.sv
else ifeq ($(fsync_mode), interrupt)
	sed -i -E "s/^(parameter bit[[:space:]]+FSYNC_STALL[[:space:]]+= )[01];/\10;/" $(MAGIA_DIR)/hw/tile/magia_tile_pkg.sv
else
	$(error unrecognized fractal sync mode (acceptable modes: stall|interrupt).)
endif
ifneq (,$(filter $(build_mode), update synth profile))
	cd $(MAGIA_DIR)														&& \
	make python_venv || true											&& \
	source setup_env.sh 												&& \
	make python_deps || true											&& \
	make bender															&& \
	make $(build_mode)-ips > $(build_mode)-ips.log mesh_dv=$(mesh_dv)	&& \
	make floonoc-patch || true											&& \
	make build-hw > build-hw.log mesh_dv=$(mesh_dv)
else
	$(error unrecognized mode (acceptable build modes: update|profile|synth).)
endif

gvsoc:
ifeq ($(target_platform), magia)
	sed -i -E "s/^[[:space:]]*N_TILES_X[[:space:]]*=[[:space:]]*[0-9]+/    N_TILES_X           = $(tiles)/" $(GVSOC_DIR)/pulp/pulp/chips/magia_base/magia_arch.py
	sed -i -E "s/^[[:space:]]*N_TILES_Y[[:space:]]*=[[:space:]]*[0-9]+/    N_TILES_Y           = $(tiles)/" $(GVSOC_DIR)/pulp/pulp/chips/magia_base/magia_arch.py
else
	$(error unrecognized platform (acceptable platform: magia).)
endif
	cd $(GVSOC_DIR)	&& \
	git submodule update --init --recursive && \
	make build TARGETS=magia-base DEBUG=1

