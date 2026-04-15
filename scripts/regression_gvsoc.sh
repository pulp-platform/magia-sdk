#!/usr/bin/env bash

testlist="test_helloworld test_fsync_levels test_fsync_rc test_fsync_diag test_mm_is test_mm_ws test_mm_os test_idma_2d test_idma_1d test_cemm_global test_mm_is_2 test_mm_os_2 test_mm_ws_2 test_fsync_lr test_gemv"
nlist="1 2 4 8 16"

rm -rf scripts/regression_output_*
eval "$(pyenv init -)"
pyenv local 3.12
python -m venv gvsoc_venv
source gvsoc_venv/bin/activate
pip install .
make gvsoc_init
for n in ${nlist}; do
    if [ ! -d scripts/regression_output_${n}_tiles/ ]
        then mkdir scripts/regression_output_${n}_tiles/
    fi
    make gvsoc tiles=${n}
    make clean build tiles=${n}
    for test in ${testlist}; do
        make run platform=gvsoc test=${test} tiles=${n} >> "scripts/regression_output_${n}_tiles/${test}.txt"
    done
done

