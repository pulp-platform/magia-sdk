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

include scripts/deps.env

CURR_DIR		?= $(shell pwd)
CMAKE_BUILDDIR  ?= $(CURR_DIR)/build
MAGIA_RTL_DIR 	?= ..
BUILD_DIR 		?= $(MAGIA_RTL_DIR)/sw/tests/$(test)
MAGIA_DIR_ABS	?= $(abspath $(MAGIA_RTL_DIR))
BUILD_DIR_ABS	?= $(MAGIA_DIR_ABS)/sw/tests/$(test)
GVSOC_DIR 		?= ./gvsoc
GVSOC_ABS_PATH	?= $(CURR_DIR)/gvsoc
BIN_ABS_PATH	?= $(CMAKE_BUILDDIR)/bin
BIN 			?= $(BUILD_DIR)/build/verif
# Prebuilt Verilator model. Built in the MAGIA repo with `make verilate mesh_dv=1`;
# the SDK never builds it. Override to use a model outside $(MAGIA_RTL_DIR).
MAGIA_VERILATOR_BIN	?= $(MAGIA_DIR_ABS)/verilator/build/obj_dir/Vmagia_tb
# Waveform a verilator run dumps under gui=1, relative to the test build dir the
# model runs in. Dumping is off otherwise and costs nothing.
VERILATOR_FST	?= $(test).fst
# Build parallelism, and threads compiled into the model. Passed to the MAGIA
# repo's verilator flow by `make MAGIA platform=verilator`.
verilator_jobs		?= 16
verilator_threads	?= 4
# Which simulator `MAGIA` and `rtl-clean` act on. `run` reads `platform` directly
# and still demands it explicitly; these two default to rtl so that invocations
# predating platform= keep working unchanged.
hw_platform	:= $(if $(platform),$(platform),rtl)
# Simulator inputs, relative to the test build dir. Same contract for questasim
# and verilator: both drive the magia_tb testbench in the MAGIA repo, so these
# mirror its own defaults.
inst_hex_name	?= build/stim_instr.txt
data_hex_name	?= build/stim_data.txt
itb_file		?= build/verif.itb
inst_entry		?= 0xCC000000
data_entry		?= 0xCC010000
boot_addr		?= 0xCC000080
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

target_platform ?= magia_v3
compiler 		?= GCC_MULTILIB
ifeq ($(target_platform), magia_v2)
ISA				?= rv32imcxgap9
else
ISA				?= rv32imac
endif
gui 			?= 0
tiles 			?= 2
spatz			?= 1
verbose			?= 0
pulp_cores		?= 8
pulp_cluster	:= $(shell [ $(pulp_cores) -gt 0 ] && echo 1 || echo 0)

LLVM_CMAKE			?= cmake
LLVM_DIR			?= llvm
LLVM_REPO			?= git@github.com:pulp-platform/llvm-project.git
LLVM_COMMIT			?= b494f2d8dde88723026db8ec16ac6c7ee1e140ca
LLVM_INSTALL_DIR	?= $(CURR_DIR)/llvm/install
LLVM_BUILD_DIR		?= $(LLVM_DIR)/llvm-project/build
LLVM_JOBS			?= 8

tiles_2 		:= $(shell echo $$(( $(tiles) * $(tiles) )))
tiles_log    	:= $(shell awk 'BEGIN { printf "%.0f", log($(tiles_2))/log(2) }')
tiles_log_real  := $(shell awk 'BEGIN { printf "%.0f", log($(tiles))/log(2) }')

GVRUN ?= $(GVSOC_DIR)/install/bin/gvrun

CMAKE ?= cmake

GVSOC_WORK_DIR ?= ./gvsoc_work
GVRUN_COMMON_ARGS ?= --work-dir $(GVSOC_WORK_DIR) --attr $(target_platform)/spatz_romfile=$(BIN_ABS_PATH)/bootrom/spatz_init.bin --trace-level=trace --trace=kill-module
GVRUN_ARGS ?= $(GVRUN_COMMON_ARGS) run
GVRUN_PROFILE_ARGS ?= $(GVRUN_COMMON_ARGS) --vcd --event=.* run
profile_tile		?=
PROFILE_TILE_ARG	= $(if $(profile_tile),--trace=tile-$(profile_tile)-idma-ctrl-mm,)
gvsoc_trace			?= 0
GVSOC_TRACE_IDS		= $(shell seq 0 $$(( $(tiles_2) - 1 )))
# NB: the trace file path is resolved relative to the GVSoC work-dir (like all.vcd), so it is a bare filename.
GVSOC_TRACE_ARG		= $(if $(filter 1,$(gvsoc_trace)),$(foreach t,$(GVSOC_TRACE_IDS),--trace='tile-$(t)-cv32-core/insn:cv32_trace_tile$(t).log'),)

# GVSOC VCD to Perfetto converter scripts (Python version kept but unused currently)
GVSOC2PERFETTO_SCRIPT  ?= scripts/gvsoc2perfetto.py
GVSOC2PERFETTO_DIR     ?= scripts/gvsoc2perfetto-rs
GVSOC2PERFETTO_BIN     ?= $(GVSOC2PERFETTO_DIR)/target/release/gvsoc2perfetto
GVSOC2PERFETTO_VCD     ?= $(GVSOC_WORK_DIR)/all.vcd
GVSOC2PERFETTO_OUT     ?= $(GVSOC_WORK_DIR)/trace.perfetto-trace
# Per-tile: CV32E40P core, light_redmule, both idma ports (frontend descriptor
# fields + real me_state/be_state FSM), Snitch+Spatz (scalar core + Spatz/ara
# vector unit). Chip-level: NoC mesh routers/network-interfaces, L2 traffic.
GVSOC2PERFETTO_INCLUDE ?= (?x) \
	tile-\d+-cv32-core\.(busy|asm|func|active_pc)$$ \
  | tile-\d+-redmule\.(busy|fsm_state)$$ \
  | tile-\d+-idma[01]\.(fe\.do_transfer_grant|me\.me_state|be\.be_state)$$ \
  | tile-\d+-snitch-spatz\.(busy|asm|func|active_pc)$$ \
  | tile-\d+-snitch-spatz\.ara\.(active|label)$$ \
  | tile-\d+-snitch-spatz\.ara\.(vfpu|vlsu|vslide)\.(active|label)$$ \
  | magia-noc\.(req|rsp|wide)_router_\d+_\d+\.(stalled_queue_\w+|req_is_write)$$ \
  | magia-noc\.(req|rsp|wide)_router_\d+_\d+\.req(_size)?$$ \
  | magia-noc\.ni_\d+_\d+\.(narrow_req|wide_req)$$ \
  | L2-mem\.(req_addr|req_size|req_is_write)$$

.PHONY: gvsoc build format run_profiling gvsoc2perfetto

# Build the Rust VCD->Perfetto converter (cargo tracks its own incremental state).
gvsoc2perfetto: $(GVSOC2PERFETTO_BIN)
$(GVSOC2PERFETTO_BIN): $(GVSOC2PERFETTO_DIR)/src/main.rs $(GVSOC2PERFETTO_DIR)/Cargo.toml $(GVSOC2PERFETTO_DIR)/Cargo.lock
	cargo build --release --manifest-path $(GVSOC2PERFETTO_DIR)/Cargo.toml

format:
	@bash scripts/ci/format-changed.sh apply

clean:
	rm -rf build/

# Note: hw-clean-all also removes .bender, which invalidates any verilator model
# built from the same checkout.
rtl-clean:
ifeq ($(hw_platform), verilator)
	cd $(MAGIA_RTL_DIR) 		&& \
	make clean-verilate
else
	cd $(MAGIA_RTL_DIR) 		&& \
	make hw-clean-all
endif
	rm -rf $(MAGIA_RTL_DIR)/sw/tests/test_*

build:
ifeq ($(tiles), )
	$(error tiles is empty!)
endif
	echo $(pulp_cores)
	@t=./targets/$(target_platform)/include/addr_map/tile_config.h; \
	printf '// Auto-generated by `make build tiles=N` -- do not edit by hand.\n// Default: 2x2 mesh. Regenerate with: make build tiles=<N>\n#define MESH_X_TILES %d\n#define MESH_Y_TILES %d\n#define MAX_SYNC_LVL %d\n#define MESH_2_POWER %d\n#define PULP_CORE_COUNT %d\n' $(tiles) $(tiles) $(tiles_log) $(tiles_log_real) $(pulp_cores) > $$t.tmp; \
	if cmp -s $$t.tmp $$t; then rm -f $$t.tmp; else mv $$t.tmp $$t; echo "Regenerated $$t"; fi
ifeq ($(compiler), LLVM)
	$(error COMING SOON!)
endif
	$(CMAKE) -DTARGET_PLATFORM=$(target_platform) -DTILES=$(tiles) -DEVAL=$(eval) -DSTALLING=$(stalling) -DFSYNC_MM=$(fsync_mm) -DIDMA_MM=$(idma_mm) -DREDMULE_MM=$(redmule_mm) -DCOMPILER=$(compiler) -DPROFILE_CMP=$(profile_cmp) -DPROFILE_CMI=$(profile_cmi) -DPROFILE_CMO=$(profile_cmo) -DPROFILE_SNC=$(profile_snc) -DSPATZ_TESTS=$(spatz) -DPULP_TESTS=$(pulp_cluster) -DPULP_CORE_COUNT=$(pulp_cores) -DPULP_CLUSTER=$(pulp_cluster) -B $(CMAKE_BUILDDIR) $(if $(filter 1,$(verbose)),--trace-expand,)
	$(CMAKE) --build $(CMAKE_BUILDDIR) $(if $(filter 1,$(verbose)),--verbose,) $(if $(test),--target $(test),) -- --no-print-directory

set_mesh:
ifeq ($(tiles), 1)
	$(eval mesh_dv=0)
endif

$(GVSOC_WORK_DIR):
	mkdir -p $(GVSOC_WORK_DIR)

# SREC -> $readmemh stimuli, replacing the parse_s19.pl | s19tomem.py pipeline.
# Single file, so rustc directly rather than a cargo project.
RUSTC			?= rustc
S19TOMEM_SRC	?= scripts/s19tomem.rs
S19TOMEM_BIN	?= $(CMAKE_BUILDDIR)/tools/s19tomem

$(S19TOMEM_BIN): $(S19TOMEM_SRC)
	mkdir -p $(dir $@)
	$(RUSTC) -O -o $@ $<

# Turn the CMake-built ELF into everything an RTL simulator needs, under
# $(MAGIA_RTL_DIR)/sw/tests/$(test)/build/: the ELF itself as `verif`, the
# $readmemh instruction/data images, and the disassembly the core tracer reads.
# Shared by platform=rtl and platform=verilator -- both drive the same magia_tb.
ifeq ($(compiler), GCC_MULTILIB)
OBJDUMP ?= riscv64-unknown-elf-objdump
else
OBJDUMP ?= riscv32-unknown-elf-objdump
endif

.PHONY: rtl_stimuli
rtl_stimuli: $(S19TOMEM_BIN)
ifndef test
	$(error Proper formatting is: make rtl_stimuli test=<test_name>)
endif
	mkdir -p $(BUILD_DIR_ABS)/build
	cp $(BIN_ABS_PATH)/$(test) $(BUILD_DIR_ABS)/build/verif
	objcopy --srec-len 1 --output-target=srec $(BIN) $(BIN).s19
	$(S19TOMEM_BIN) $(BIN).s19 $(BUILD_DIR_ABS)/build/stim_instr.txt $(BUILD_DIR_ABS)/build/stim_data.txt
	$(OBJDUMP) -d -S -Mmarch=$(ISA) $(BIN) > $(BIN).dump
	$(OBJDUMP) -d -l -s -Mmarch=$(ISA) $(BIN) > $(BIN).objdump
	python3 scripts/objdump2itb.py $(BIN).objdump > $(BIN).itb

run: set_mesh $(GVSOC_WORK_DIR)
	@echo 'Magia is available at https://github.com/pulp-platform/MAGIA.git'
	@echo 'please run "source setup_env.sh" in the magia folder before running this script'
	@echo 'and make sure the risc-v objdump binary is visible on path using "which riscv32-unknown-elf-objdump".'
ifndef test
	$(error Proper formatting is: make run test=<test_name> platform=rtl|verilator|gvsoc)
endif
ifeq (,$(wildcard $(CMAKE_BUILDDIR)/bin/$(test)))
	$(error No test found with name: $(test))
endif
ifndef platform
	$(error Proper formatting is: make run test=<test_name> platform=rtl|verilator|gvsoc)
endif
ifeq ($(platform), gvsoc)
	$(GVRUN) --target=$(target_platform):n_tiles_x=$(tiles),n_tiles_y=$(tiles),nb_pulp_cores=$(pulp_cores) --param binary=$(BIN_ABS_PATH)/$(test) $(GVRUN_ARGS)
else ifeq ($(platform), rtl)
	$(MAKE) rtl_stimuli test=$(test)
	cd $(BUILD_DIR_ABS)													&& \
	cp -sf "$(MAGIA_DIR_ABS)/sim/modelsim.ini" modelsim.ini    			&& \
	ln -sfn "$(MAGIA_DIR_ABS)/sim/work" work
	cd $(MAGIA_RTL_DIR) 												&& \
	make run test=$(test) gui=$(gui) mesh_dv=$(mesh_dv) fast_sim=$(fast_sim)
else ifeq ($(platform), verilator)
	@test -x "$(MAGIA_VERILATOR_BIN)" || {								\
	  echo "error: no Verilator model at $(MAGIA_VERILATOR_BIN)" >&2;	\
	  echo "       build it in the MAGIA repo first:" >&2;				\
	  echo "         make verilate core=CV32E40P mesh_dv=1" >&2;		\
	  exit 1; }
	$(MAKE) rtl_stimuli test=$(test)
# No modelsim.ini/work symlinks: the verilated model is self-contained. Run it
# straight rather than via the MAGIA repo's `make verilate-run`, whose `all`
# prerequisite would try to recompile the test from sources that only exist for
# tests living in that repo.
	set -o pipefail												 	 && \
	cd $(BUILD_DIR_ABS)												 	 && \
	"$(MAGIA_VERILATOR_BIN)"											\
	  +INST_HEX=$(inst_hex_name)										\
	  +DATA_HEX=$(data_hex_name)										\
	  +INST_ENTRY=$(inst_entry)											\
	  +DATA_ENTRY=$(data_entry)											\
	  +BOOT_ADDR=$(boot_addr)											\
	  +itb_file=$(itb_file)												\
	  $(if $(filter 1,$(gui)),+FST=$(VERILATOR_FST),)					\
	  2>&1 | tee transcript_verilator
ifeq ($(gui), 1)
	@echo ''
	@echo 'Waveform: $(BUILD_DIR_ABS)/$(VERILATOR_FST)'
	@echo '  gtkwave $(BUILD_DIR_ABS)/$(VERILATOR_FST)'
	@echo '  surfer  $(BUILD_DIR_ABS)/$(VERILATOR_FST)'
endif
else
	$(error Only rtl, verilator and gvsoc are supported as platforms.)
endif

run_profiling: set_mesh $(GVSOC_WORK_DIR) $(GVSOC2PERFETTO_BIN)
ifndef test
	$(error Proper formatting is: make run_profiling test=<test_name>)
endif
ifeq (,$(wildcard $(CMAKE_BUILDDIR)/bin/$(test)))
	$(error No test found with name: $(test))
endif
	$(GVRUN) --target $(target_platform) --param binary=$(BIN_ABS_PATH)/$(test) $(GVRUN_PROFILE_ARGS) $(PROFILE_TILE_ARG) $(GVSOC_TRACE_ARG)
	$(GVSOC2PERFETTO_BIN) $(GVSOC2PERFETTO_VCD) \
		-o $(GVSOC2PERFETTO_OUT) \
		--state-map 'fsm_state=0:idle,1:preload,2:routine,3:storing,4:finished,5:acknowledge' \
		--state-map 'me_state=0:idle,1:decomposing' \
		--state-map 'be_state=0:idle,1:active' \
		--rename 'ara=vfu' \
		--rename 'label=instructions' \
		--rename 'active_pc=pc' \
		--split-asm \
		--symbolize $(CMAKE_BUILDDIR)/bin/$(test).s \
		--stats \
		--include '$(GVSOC2PERFETTO_INCLUDE)'
	rm -f -- $(GVSOC2PERFETTO_VCD)

debug_profiling: $(GVSOC_WORK_DIR) $(GVSOC2PERFETTO_BIN)
ifndef test
	$(error Proper formatting is: make debug_profiling test=<test_name>)
endif
	$(GVSOC2PERFETTO_BIN) $(GVSOC2PERFETTO_VCD) \

		-o $(GVSOC2PERFETTO_OUT) \
		--state-map 'fsm_state=0:idle,1:preload,2:routine,3:storing,4:finished,5:acknowledge' \
		--state-map 'me_state=0:idle,1:decomposing' \
		--state-map 'be_state=0:idle,1:active' \
		--rename 'ara=vfu' \
		--rename 'label=instructions' \
		--rename 'active_pc=pc' \
		--split-asm \
		--symbolize $(CMAKE_BUILDDIR)/bin/$(test).s \
		--stats \
		--include '$(GVSOC2PERFETTO_INCLUDE)'
	rm -f -- $(GVSOC2PERFETTO_VCD)

# Hardware build command, the only part of `make MAGIA` that differs per simulator.
# Recursively expanded on purpose: set_mesh rewrites mesh_dv while the target runs.
ifeq ($(hw_platform), verilator)
HW_BUILD_CMD = make verilate > verilate.log mesh_dv=$(mesh_dv) VERILATOR_JOBS=$(verilator_jobs) VERILATOR_THREADS=$(verilator_threads)
else
HW_BUILD_CMD = make build-hw > build-hw.log mesh_dv=$(mesh_dv) fast_sim=$(fast_sim)
endif

MAGIA: set_mesh
ifeq (,$(filter $(hw_platform), rtl verilator))
	$(error Only rtl and verilator can be built with `make MAGIA` (got platform=$(hw_platform)).)
endif
# The MAGIA verilator flow is mesh-only and CV32E40P-only (verilator/verilator.mk).
# tiles=1 is rejected because set_mesh turns it into mesh_dv=0, and magia_v1
# because it seds the core to CV32E40X further down.
ifeq ($(hw_platform), verilator)
ifeq ($(tiles), 1)
	$(error platform=verilator requires mesh_dv=1, but tiles=1 forces mesh_dv=0: there is no single-tile verilator target.)
endif
ifneq ($(mesh_dv), 1)
	$(error platform=verilator requires mesh_dv=1: there is no single-tile verilator target.)
endif
ifeq ($(target_platform), magia_v1)
	$(error platform=verilator does not support magia_v1: it selects CV32E40X, whose hierarchical core traces need per-tile filenames resolved at run time.)
endif
endif
ifeq ($(shell expr $(tiles_2) \> 256), 1)
	$(eval tiles_2=256)
endif
ifeq ($(target_platform), magia_v1)
	sed -i -E 's/^(num_cores[[:space:]]*\?=[[:space:]]*)[0-9]+/\1$(tiles_2)/' $(MAGIA_RTL_DIR)/Makefile
	sed -i -E 's/^(core[[:space:]]*\?=[[:space:]]*)CV32E40P/\1CV32E40X/' $(MAGIA_RTL_DIR)/Makefile
else ifeq ($(target_platform), magia_v2)
	sed -i -E 's/^(num_cores[[:space:]]*\?=[[:space:]]*)[0-9]+/\1$(tiles_2)/' $(MAGIA_RTL_DIR)/Makefile
	sed -i -E 's/^(core[[:space:]]*\?=[[:space:]]*)CV32E40X/\1CV32E40P/' $(MAGIA_RTL_DIR)/Makefile
else ifeq ($(target_platform), magia_v3)
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
	curl --proto '=https' --tlsv1.2 https://pulp-platform.github.io/bender/init -sSf | sh -s -- --local && \
	export PATH=$$(pwd):$$PATH											&& \
	python -m pip install --upgrade "setuptools<81"						&& \
	make vsim-scripts > vsim-scripts.log mesh_dv=$(mesh_dv)	&& \
	make floonoc-patch || true											&& \
	$(HW_BUILD_CMD)
else
	$(error unrecognized mode (acceptable build modes: update|profile|synth).)
endif

gvsoc:
ifeq ($(target_platform), magia_v2)
	sed -i -E "s/^[[:space:]]*N_TILES_X[[:space:]]*=[[:space:]]*[0-9]+/    N_TILES_X           = $(tiles)/" $(GVSOC_DIR)/pulp/pulp/chips/magia_v2/arch.py
	sed -i -E "s/^[[:space:]]*N_TILES_Y[[:space:]]*=[[:space:]]*[0-9]+/    N_TILES_Y           = $(tiles)/" $(GVSOC_DIR)/pulp/pulp/chips/magia_v2/arch.py
ifeq ($(spatz), 1)
	sed -i 's/^\([[:space:]]*SPATZ_ENABLE[[:space:]]*=[[:space:]]*\)False/\1True/' $(GVSOC_DIR)/pulp/pulp/chips/magia_v2/arch.py
else
	sed -i 's/^\([[:space:]]*SPATZ_ENABLE[[:space:]]*=[[:space:]]*\)True/\1False/' "$(GVSOC_DIR)/pulp/pulp/chips/magia_v2/arch.py"
endif
	cd $(GVSOC_DIR)	&& \
	make build TARGETS=magia_v2
else ifeq ($(target_platform), magia_v3)
	cd $(GVSOC_DIR)	&& \
	make build TARGETS="magia_v3:n_tiles_x=$(tiles),n_tiles_y=$(tiles),nb_pulp_cores=$(pulp_cores)"
else
	$(error unrecognized platform (acceptable platforms: magia_v2, magia_v3).)
endif

# Pinned commits for FondazioneChipsIT/gvsoc, gvsoc-core, pulp, engine, gvrun live in scripts/deps.env.
gvsoc_init:
	git clone https://github.com/FondazioneChipsIT/gvsoc.git || true
	cd $(GVSOC_DIR) && \
	git fetch origin $(GVSOC_COMMIT) && \
	git checkout $(GVSOC_COMMIT) && \
	git submodule update --init --recursive && \
	cd core && \
	git fetch origin $(GVSOC_CORE_COMMIT) && \
	git checkout $(GVSOC_CORE_COMMIT) && \
	cd ../pulp && \
	git fetch origin $(GVSOC_PULP_COMMIT) && \
	git checkout $(GVSOC_PULP_COMMIT) && \
	cd ../engine && \
	git fetch origin $(GVSOC_ENGINE_COMMIT) && \
	git checkout $(GVSOC_ENGINE_COMMIT) && \
	cd ../gvrun && \
	git fetch origin $(GVSOC_GVRUN_COMMIT) && \
	git checkout $(GVSOC_GVRUN_COMMIT)

# Set TORCH=1 to also install the "gemm" extra (torch, CPU build via uv).
TORCH ?= 0
ifeq ($(TORCH),1)
PIP_EXTRAS := [gemm]
endif

gvsoc_uv:
	uv venv --python 3.12 gvsoc_venv && \
	source gvsoc_venv/bin/activate && \
	uv pip install .$(PIP_EXTRAS)

gvsoc_venv:
	eval "$(pyenv init -)" && \
	pyenv local 3.12 && \
	python -m venv gvsoc_venv && \
	source gvsoc_venv/bin/activate && \
	pip install .$(PIP_EXTRAS)

llvm:
	mkdir -p $(LLVM_DIR)
	if [ ! -d "$(LLVM_DIR)/llvm-project/.git" ]; then \
		cd $(LLVM_DIR) && git clone $(LLVM_REPO); \
	fi
	cd $(LLVM_DIR)/llvm-project && \
	git checkout $(LLVM_COMMIT) && \
	git submodule update --init --recursive --jobs=$(LLVM_JOBS) .
	mkdir -p $(LLVM_INSTALL_DIR)
	cd $(LLVM_DIR)/llvm-project && mkdir -p build && cd build && \
	$(LLVM_CMAKE) \
		-DCMAKE_INSTALL_PREFIX=$(LLVM_INSTALL_DIR) \
		-DCMAKE_CXX_COMPILER=${CXX} \
		-DCMAKE_C_COMPILER=${CC} \
		-DLLVM_OPTIMIZED_TABLEGEN=True \
		-DLLVM_ENABLE_PROJECTS="clang;lld" \
		-DLLVM_TARGETS_TO_BUILD="RISCV" \
		-DLLVM_DEFAULT_TARGET_TRIPLE=riscv32-unknown-elf \
		-DLLVM_ENABLE_LLD=False \
		-DLLVM_APPEND_VC_REV=ON \
		-DCMAKE_BUILD_TYPE=Release \
		../llvm && \
	make -j$(LLVM_JOBS) all && \
	make install
