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

CMAKE_BUILDDIR  	?= $(CURR_DIR)/build
MAGIA_RTL_DIR 		?= ..
BUILD_DIR 		?= $(MAGIA_RTL_DIR)/sw/tests/$(test)
GVSOC_DIR 		?= ./gvsoc
CURR_DIR		?= $(shell pwd)
GVSOC_ABS_PATH	?= $(CURR_DIR)/gvsoc
BIN_ABS_PATH	?= $(CMAKE_BUILDDIR)/bin
BIN 			?= $(BUILD_DIR)/build/verif
build_mode		?= update
fsync_mode		?= stall
mesh_dv			?= 1
fast_sim		?= 0
eval			?= 0
stalling		?= 0
fsync_mm		?= 1
idma_mm			?= 1
redmule_mm		?= 1
profile_cmp		?= 0
profile_cmi		?= 0
profile_cmo		?= 0
profile_snc		?= 0

target_platform ?= magia_v2
compiler 		?= GCC_PULP
ISA				?= rv32imcxgap9
gui 			?= 0
tiles 			?= 2

# FlatAttention pipeline parameters
s_size           ?= 16
d_size           ?= 8
seed             ?= 42
flatatt_platform ?= gvsoc
data_tiling      ?= 0

FLATATT_TEST = $(if $(filter 1,$(data_tiling)),test_flatatt,test_flatatt_no_dt)

tiles_2 		:= $(shell echo $$(( $(tiles) * $(tiles) )))
tiles_log    	:= $(shell awk 'BEGIN { printf "%.0f", log($(tiles_2))/log(2) }')
tiles_log_real  := $(shell awk 'BEGIN { printf "%.0f", log($(tiles))/log(2) }')

GVRUN ?= $(GVSOC_DIR)/install/bin/gvrun
GVRUN_ARGS ?= --work-dir $(GVSOC_ABS_PATH)/Documents/test --attr magia_v2/n_tiles_x=$(tiles) --attr magia_v2/n_tiles_y=$(tiles) --trace-level=trace run --trace=kill-module

.PHONY: gvsoc build flatatt flatatt-gen flatatt-build flatatt-run flatatt-ci

clean:
	rm -rf build/

rtl-clean:
	cd $(MAGIA_RTL_DIR) 		&& \
	make hw-clean-all
	rm -rf $(MAGIA_RTL_DIR)/sw/tests/test_*

build:
ifeq ($(tiles), )
	$(error tiles is empty!)
endif
	sed -i -E 's/^#define MESH_([XY])_TILES[[:space:]]*[0-9]+/#define MESH_\1_TILES $(tiles)/' ./targets/$(target_platform)/include/addr_map/tile_addr_map.h
	sed -i -E 's/^(#define MAX_SYNC_LVL[[:space:]]*)[0-9]+/\1$(tiles_log)/' ./targets/$(target_platform)/include/addr_map/tile_addr_map.h
	sed -i -E 's/^(#define MESH_2_POWER[[:space:]]*)[0-9]+/\1$(tiles_log_real)/' ./targets/$(target_platform)/include/addr_map/tile_addr_map.h
ifeq ($(compiler), LLVM)
	$(error COMING SOON!)
endif
ifeq ($(compiler), GCC_PULP)
	sed -i -E 's/^add_subdirectory\(flatatt\)/#&/' ./tests/magia/mesh/CMakeLists.txt
	sed -i -E 's/^add_subdirectory\(flatatt_no_data_tiling\)/#&/' ./tests/magia/mesh/CMakeLists.txt
	sed -i -E 's/^#include "utils\/attention_utils.h"/\/\/&/' ./targets/$(target_platform)/include/tile.h
endif
ifeq ($(compiler), GCC_MULTILIB)
	sed -i -E 's/^#add_subdirectory\(flatatt\)/add_subdirectory\(flatatt\)/' ./tests/magia/mesh/CMakeLists.txt
	sed -i -E 's/^#add_subdirectory\(flatatt_no_data_tiling\)/add_subdirectory\(flatatt_no_data_tiling\)/' ./tests/magia/mesh/CMakeLists.txt
	sed -i -E 's/^\/\/#include "utils\/attention_utils.h"/#include "utils\/attention_utils.h"/' ./targets/$(target_platform)/include/tile.h
endif
	cmake -DTARGET_PLATFORM=$(target_platform) -DEVAL=$(eval) -DSTALLING=$(stalling) -DFSYNC_MM=$(fsync_mm) -DIDMA_MM=$(idma_mm) -DREDMULE_MM=$(redmule_mm) -DCOMPILER=$(compiler) -DPROFILE_CMP=$(profile_cmp) -DPROFILE_CMI=$(profile_cmi) -DPROFILE_CMO=$(profile_cmo) -DPROFILE_SNC=$(profile_snc) -B $(CMAKE_BUILDDIR) --trace-expand
	cmake --build $(CMAKE_BUILDDIR) --verbose

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
ifeq (,$(wildcard $(CMAKE_BUILDDIR)/bin/$(test)))
	$(error No test found with name: $(test))
endif
ifndef platform
	$(error Proper formatting is: make run test=<test_name> platform=rtl|gvsoc)
endif
ifeq ($(platform), gvsoc)
	$(GVRUN) --target magia_v2 --param binary=$(BIN_ABS_PATH)/$(test) $(GVRUN_ARGS)
else ifeq ($(platform), rtl)
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && mkdir -p build
	cp ./build/bin/$(test) $(BUILD_DIR)/build/verif
	objcopy --srec-len 1 --output-target=srec $(BIN) $(BIN).s19
	scripts/parse_s19.pl $(BIN).s19 > $(BIN).txt
	python3 scripts/s19tomem.py $(BIN).txt $(BUILD_DIR)/build/stim_instr.txt $(BUILD_DIR)/build/stim_data.txt
	cd $(BUILD_DIR)													&& \
	cp -sf ../../../sim/modelsim.ini modelsim.ini    				&& \
	ln -sfn ../../../sim/work work
	riscv32-unknown-elf-objdump -d -S -Mmarch=$(ISA) $(BIN) > $(BIN).dump
	riscv32-unknown-elf-objdump -d -l -s -Mmarch=$(ISA) $(BIN) > $(BIN).objdump
	python3 scripts/objdump2itb.py $(BIN).objdump > $(BIN).itb
	cd $(MAGIA_RTL_DIR) 												&& \
	make run test=$(test) gui=$(gui) mesh_dv=$(mesh_dv)
else
	$(error Only rtl and gvsoc are supported as platforms.)
endif

MAGIA: set_mesh
ifeq ($(shell expr $(tiles_2) \> 256), 1)
	$(eval tiles_2=256)
endif
ifeq ($(target_platform), magia_v1)
	sed -i -E 's/^(num_cores[[:space:]]*\?=[[:space:]]*)[0-9]+/\1$(tiles_2)/' $(MAGIA_RTL_DIR)/Makefile
	sed -i -E 's/^(core[[:space:]]*\?=[[:space:]]*)CV32E40P/\1CV32E40X/' $(MAGIA_RTL_DIR)/Makefile
else ifeq ($(target_platform), magia_v2)
	sed -i -E 's/^(num_cores[[:space:]]*\?=[[:space:]]*)[0-9]+/\1$(tiles_2)/' $(MAGIA_RTL_DIR)/Makefile
	sed -i -E 's/^(core[[:space:]]*\?=[[:space:]]*)CV32E40X/\1CV32E40P/' $(MAGIA_RTL_DIR)/Makefile
else
	$(error unrecognized platform (acceptable platform: magia).)
endif
ifneq ($(tiles), 1)
	sed -i -E 's/^(  localparam int unsigned N_TILES_[XY][[:space:]]*=[[:space:]]*)[0-9]+;/\1$(tiles);/' $(MAGIA_RTL_DIR)/hw/mesh/magia_pkg.sv
endif
ifeq ($(fsync_mode), stall)
	sed -i -E 's/(FSYNC_STALL[[:space:]]=[[:space:]])[0-9]+/\11/' $(MAGIA_RTL_DIR)/hw/tile/magia_tile_pkg.sv
else ifeq ($(fsync_mode), interrupt)
	sed -i -E 's/(FSYNC_STALL[[:space:]]=[[:space:]])[0-9]+/\10/' $(MAGIA_RTL_DIR)/hw/tile/magia_tile_pkg.sv
else
	$(error unrecognized fractal sync mode (acceptable modes: stall|interrupt).)
endif
ifneq (,$(filter $(build_mode), update synth profile))
	cd $(MAGIA_RTL_DIR)														&& \
	make python_venv || true											&& \
	source setup_env.sh 												&& \
	make python_deps || true											&& \
	python -m pip install --upgrade "setuptools<81"						&& \
	make bender															&& \
	make $(build_mode)-ips > $(build_mode)-ips.log mesh_dv=$(mesh_dv)	&& \
	make floonoc-patch || true											&& \
	make build-hw > build-hw.log mesh_dv=$(mesh_dv) fast_sim=$(fast_sim)
else
	$(error unrecognized mode (acceptable build modes: update|profile|synth).)
endif

gvsoc:
ifeq ($(target_platform), magia_v2)
	sed -i -E "s/^[[:space:]]*N_TILES_X[[:space:]]*=[[:space:]]*[0-9]+/    N_TILES_X           = $(tiles)/" $(GVSOC_DIR)/pulp/pulp/chips/magia_v2/arch.py
	sed -i -E "s/^[[:space:]]*N_TILES_Y[[:space:]]*=[[:space:]]*[0-9]+/    N_TILES_Y           = $(tiles)/" $(GVSOC_DIR)/pulp/pulp/chips/magia_v2/arch.py
else
	$(error unrecognized platform (acceptable platform: magia_v2).)
endif
	cd $(GVSOC_DIR)	&& \
	make build TARGETS=magia_v2

# github.com/gvsoc/gvsoc: 5f91fcc commit - master on 24/04/2026
# github.com/gvsoc/gvsoc-core: 760f25e commit - master on 24/04/2026
# github.com/gvsoc/gvsoc-pulp: 17f9e7a commit - master on 24/04/2026
gvsoc_init:
	git clone https://github.com/gvsoc/gvsoc.git || true
	cd $(GVSOC_DIR) && \
	git fetch origin 5f91fcc2e5923993adaaf5aad3365b690da19da2 && \
	git checkout 5f91fcc2e5923993adaaf5aad3365b690da19da2 && \
	git submodule update --init --recursive && \
	cd core && \
	git fetch origin 760f25eacb135dac60acd61eea5d6f1e3611192d && \
	git checkout 760f25eacb135dac60acd61eea5d6f1e3611192d && \
	cd ../pulp && \
	git fetch origin 17f9e7a56f7ccf0bf7d13af0e68b6c0c129f8555 && \
	git checkout 17f9e7a56f7ccf0bf7d13af0e68b6c0c129f8555

gvsoc_venv:
	eval "$(pyenv init -)" && \
	pyenv local 3.12 && \
	python -m venv gvsoc_venv && \
	source gvsoc_venv/bin/activate && \
	pip install .

# ─── FlatAttention pipeline ──────────────────────────────────────────
# Usage:
#   make flatatt                                    # no data tiling (default)
#   make flatatt data_tiling=1                      # with data tiling (original)
#   make flatatt tiles=4 s_size=32 d_size=16        # 4x4, larger problem
#   make flatatt flatatt_platform=rtl               # RTL simulator
#   make flatatt-gen s_size=32 d_size=16            # regenerate golden only

flatatt-gen:
	python3 tests/magia/mesh/flatatt/gen_golden.py \
		--s-size $(s_size) --d-size $(d_size) --mesh $(tiles) --seed $(seed)
	mkdir -p tests/magia/mesh/flatatt_no_data_tiling/include
	cp tests/magia/mesh/flatatt/include/test.h tests/magia/mesh/flatatt_no_data_tiling/include/test.h

flatatt-build:
	$(MAKE) clean
	$(MAKE) build target_platform=magia_v2 tiles=$(tiles) compiler=GCC_MULTILIB eval=$(eval)

flatatt-run:
	$(MAKE) run test=$(FLATATT_TEST) platform=$(flatatt_platform) tiles=$(tiles)

flatatt:
	$(MAKE) flatatt-gen tiles=$(tiles) s_size=$(s_size) d_size=$(d_size) seed=$(seed)
	$(MAKE) flatatt-build tiles=$(tiles) eval=$(eval)
ifeq ($(flatatt_platform), gvsoc)
	$(MAKE) gvsoc tiles=$(tiles)
endif
	$(MAKE) flatatt-run tiles=$(tiles) flatatt_platform=$(flatatt_platform)

# ─── FlatAttention CI matrix ─────────────────────────────────────────
# Runs all (tiles, S_size, D_size) combinations.
# GVSoC is rebuilt once per tile count to avoid redundant rebuilds.
#   make flatatt-ci                          # all 27 combos on gvsoc
#   make flatatt-ci flatatt_platform=rtl     # all 27 combos on RTL

FLATATT_TILES  := 2 4 8
FLATATT_D      := 8 16 32
FLATATT_S_MULT := 4 8 16

flatatt-ci:
	@failed=""; total=0; passed=0; \
	for t in $(FLATATT_TILES); do \
		echo ""; \
		echo "====== Building GVSoC for tiles=$$t ======"; \
		$(MAKE) gvsoc tiles=$$t || { echo "GVSOC BUILD FAILED for tiles=$$t"; exit 1; }; \
		for s_mult in $(FLATATT_S_MULT); do \
			s=$$((s_mult * t)); \
			for d in $(FLATATT_D); do \
				total=$$((total + 1)); \
				echo ""; \
				echo "====== [$${total}] tiles=$$t S=$$s D=$$d ======"; \
				if $(MAKE) flatatt-gen tiles=$$t s_size=$$s d_size=$$d seed=$(seed) && \
				   $(MAKE) flatatt-build tiles=$$t eval=$(eval) && \
				   $(MAKE) flatatt-run  tiles=$$t flatatt_platform=$(flatatt_platform); then \
					passed=$$((passed + 1)); \
					echo "PASS: tiles=$$t S=$$s D=$$d"; \
				else \
					failed="$$failed tiles=$$t,S=$$s,D=$$d"; \
					echo "FAIL: tiles=$$t S=$$s D=$$d"; \
				fi; \
			done; \
		done; \
	done; \
	echo ""; \
	echo "====== Summary: $$passed/$$total passed ======"; \
	if [ -n "$$failed" ]; then \
		echo "Failed:$$failed"; \
		exit 1; \
	fi
