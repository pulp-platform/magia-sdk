#!/usr/bin/env bash
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0


testlist="test_helloworld test_fsync_levels test_fsync_rc test_fsync_diag test_mm_is test_mm_ws test_mm_os test_idma_2d test_idma_1d test_cemm_global test_mm_is_2 test_mm_os_2 test_mm_ws_2 test_fsync_lr test_gemv"
testlist_spatz="fft_fs hello_spatz onnx_add onnx_averagepool onnx_batchnorm onnx_ceil onnx_clip onnx_div onnx_exp onnx_gemm onnx_globalaveragepool onnx_globalmaxpool onnx_groupnorm onnx_hardsigmoid onnx_hardswish onnx_instancenorm onnx_layernorm onnx_maxpool onnx_relu onnx_sigmoid onnx_softmax onnx_sub onnx_swish"
nlist="1 2 4 8 16"
spatz=1
pulp_cores=0

rm -rf scripts/regression_output_*
# eval "$(pyenv init -)"
# pyenv local 3.12
# python -m venv gvsoc_venv
# source gvsoc_venv/bin/activate
# pip install .
# make gvsoc_init
for n in ${nlist}; do
    if [ ! -d scripts/regression_output_${n}_tiles/ ]; then 
        mkdir scripts/regression_output_${n}_tiles/
    fi
    if [ ${spatz} ] && [ ! -d scripts/regression_output_spatz_${n}_tiles/ ]; then 
        mkdir scripts/regression_output_spatz_${n}_tiles/
    fi
    make gvsoc tiles=${n}
    make clean build tiles=${n} spatz=${spatz} pulp_cores=${pulp_cores} LLVM_INSTALL_DIR=/srv/home/alberto.dequino/MAGIA/magia-sdk/llvm/install/
    for test in ${testlist}; do
        make run platform=gvsoc test=${test} tiles=${n} >> "scripts/regression_output_${n}_tiles/${test}.txt"
    done
    if [ ${spatz} ] && [ !${pulp_cores} ]; then
        for test in ${testlist_spatz}; do
            make run platform=gvsoc test=${test} tiles=${n} >> "scripts/regression_output_spatz_${n}_tiles/${test}.txt"
        done
    fi
done

