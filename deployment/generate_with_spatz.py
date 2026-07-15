import os
import re
import shutil
from pathlib import Path
from argparse import ArgumentParser
import logging
import onnx
import onnx_graphsurgeon as gs
from typing import Sequence
import numpy as np
import coloredlogs
from Deeploy.Logging import DEFAULT_LOGGER, DEFAULT_FMT

from generate import (load_npz, generate_test_header, generate_network_header,
                      generate_network_source, allocator_patch, copyright_comment,
                      defaultScheduler)

from MagiaDeeployTarget.Deployer import MagiaDeployer
from MagiaDeeployTarget.Platform import MagiaPlatform, MagiaOptimizer
from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions import DataTypes

def normalize_spatz_types(code: str) -> str:
    return code.replace("float16_t", "float16")

def extract_operators(network_source: str, format: str, arch: str) -> list:
    pattern = re.compile(rf"MAGIA_([a-z0-9]+)_{format}_{arch}\s*\(")
    return sorted(set(pattern.findall(network_source)))

def add_kernel_includes(code: str, operators: list, format: str, arch: str) -> str:
    includes = "".join(f'#include "{op}_{format}_{arch}.h"\n' for op in operators)
    return code.replace('#include <stdint.h>\n', f'#include <stdint.h>\n{includes}', 1)

def split_test_header_definitions(header: str) -> tuple[str, str]:
    header_lines = []
    source_lines = [copyright_comment('//'), "", '#include "data.h"', ""]

    for line in header.splitlines():
        stripped = line.strip()

        # Identify the lines containing variable or array definitions (e.g., float16 input0[...] = { ... };)
        if " = " in line and stripped.endswith(";"):
            declaration = line.split("=", 1)[0].rstrip()
            header_lines.append(f"extern {declaration};")
            source_lines.append(line)
            continue

        header_lines.append(line)

    return "\n".join(header_lines) + "\n", "\n".join(source_lines) + "\n"

def generate_cmakelist_with_spatz(test: str, operators: list, format: str, arch: str) -> str:
    kernels_root = "${CMAKE_CURRENT_SOURCE_DIR}/../../../kernels"
    kernel_common_path = f"{kernels_root}/common"

    def kernel_root(op):
        return f"{kernels_root}/{op}/{format}/{arch}"

    task_sources  = [f"{kernel_root(op)}/spatz_task/{op}_{format}_{arch}_task.c" for op in operators]
    host_sources  = [f"{kernel_root(op)}/src/{op}_{format}_{arch}.c" for op in operators]
    include_paths = [f"{kernel_root(op)}/include" for op in operators]
    first_task    = f"{operators[0]}_{format}_{arch}_task"

    text = copyright_comment('#')
    text += "\n"
    text += f"set(TEST_NAME {test})\n"
    text += "\n"

    text += "# Step 1: Compile the Spatz task and generate the C header\n"
    text += "add_spatz_task(\n"
    text += "    TEST_NAME ${TEST_NAME}\n"
    text += "    TASK_SOURCES\n"
    for src in task_sources:
        text += f"        {src}\n"
    text += f"    FIRST_TASK_NAME {first_task}\n"
    text += "    INCLUDE_DIRS\n"
    text += f"        {kernel_common_path}\n"
    for inc in include_paths:
        text += f"        {inc}\n"
    text += "        ${CMAKE_CURRENT_SOURCE_DIR}/include\n"
    text += ")\n\n"

    text += "# Step 2: Compile the CV32 executable and embed the Spatz binary\n"
    text += "add_cv32_executable_with_spatz(\n"
    text += "    TARGET_NAME ${TEST_NAME}\n"
    text += "    SPATZ_HEADER ${SPATZ_HEADER}\n"
    text += "    SOURCES\n"
    for src in host_sources:
        text += f"        {src}\n"
    text += "        src/network.c\n"
    text += "        src/main.c\n"
    text += "        src/data.c\n"
    text += "    INCLUDE_DIRS\n"
    for inc in include_paths:
        text += f"        {inc}\n"
    text += f"        {kernel_common_path}\n"
    text += "        ${CMAKE_CURRENT_SOURCE_DIR}/include\n"
    text += ")\n\n"

    return text


def main(test) -> None:

    print(f"test: {test}")

    test = '_'.join(test.split('/')[-3:])
    operand, format, arch = test.split("_")
    src_dir = Path("deployment") / "tests" / operand / format / arch
    dst_dir = Path("tests") / "spatz_on_magia" / ("deeploy_" + test)

    # load inputs, outputs, and network
    logger.debug("loading inputs and outputs data")
    inputs = load_npz(src_dir / 'inputs.npz')
    outputs = load_npz(src_dir / 'outputs.npz')

    logger.debug("loading onnx network")
    onnx_graph = onnx.load_model(src_dir / 'network.onnx')
    graph = gs.import_onnx(onnx_graph)

    # get input types from inputs numpy arrays
    inputs_type = {}
    for i, array in enumerate(inputs):
        _type = f'{np.dtype(array.dtype).name}_t'
        inputs_type[f"input_{i}"] = PointerClass(getattr(DataTypes, _type))

    # Magia deployer
    deployer = MagiaDeployer(
        graph=graph,
        deploymentPlatform=MagiaPlatform(),
        inputTypes=inputs_type,
        loweringOptimizer=MagiaOptimizer,
        scheduler=defaultScheduler,
        name="DeeployNetwork",
        default_channels_first=True,
        deeployStateDir="states",
    )

    # run deployment process to be ready to generate code
    logger.debug("running deployment process")
    deployer.prepare()

    # create destination folders
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_inc_dir = dst_dir / 'include'
    dst_inc_dir.mkdir(parents=True, exist_ok=True)
    dst_src_dir = dst_dir / 'src'
    dst_src_dir.mkdir(parents=True, exist_ok=True)

    # prepare formatting code command
    clang_format = "{BasedOnStyle: llvm, IndentWidth: 4, ColumnLimit: 80}"
    clang_cmd = lambda path: f'clang-format -i --style="{clang_format}" {path}'

    # header for data inputs and outputs
    data_header_path = dst_inc_dir / 'data.h'
    data_source_path = dst_src_dir / 'data.c'
    logger.debug(f"generating {data_header_path}")
    data_header = generate_test_header(inputs, outputs)
    data_header = normalize_spatz_types(data_header)
    data_header, data_source = split_test_header_definitions(data_header)
    with open(data_header_path, "w") as f:
        f.write(data_header)
    with open(data_source_path, "w") as f:
        f.write(data_source)
    os.system(clang_cmd(data_header_path))

    # source for network (generated first so we can discover which kernels it calls)
    network_source_path = dst_src_dir / 'network.c'
    logger.debug(f"generating {network_source_path}")
    network_source = generate_network_source(deployer)
    network_source = allocator_patch(network_source, inputs, outputs)
    network_source = normalize_spatz_types(network_source)

    # discover the operators actually used and validate their kernels exist
    operators = extract_operators(network_source, format, arch)
    if not operators:
        raise RuntimeError(f"No MAGIA_<op>_{format}_{arch}(...) calls found in the generated network.")
    missing = [op for op in operators if not (Path("kernels") / op / format / arch).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing kernels for operators {missing} (looked in kernels/<op>/{format}/{arch}).")
    logger.info(f"network operators ({len(operators)}): {operators}")

    with open(network_source_path, "w") as f:
        f.write(network_source)
    os.system(clang_cmd(network_source_path))

    # header for network
    network_header_path = dst_inc_dir / 'network.h'
    logger.debug(f"generating {network_header_path}")
    network_header = generate_network_header(deployer)
    network_header = add_kernel_includes(network_header, operators, format, arch)
    network_header = allocator_patch(network_header, inputs, outputs)
    network_header = normalize_spatz_types(network_header)
    with open(network_header_path, "w") as f:
        f.write(network_header)
    os.system(clang_cmd(network_header_path))

    # main
    deployment_root = Path(__file__).parent
    main_src_path = deployment_root / 'main.c'
    main_dst_path = dst_src_dir / 'main.c'
    shutil.copyfile(main_src_path, main_dst_path)
    os.system(clang_cmd(main_dst_path))

    # CMakeLists.txt
    cmakelists_path = dst_dir / 'CMakeLists.txt'
    logger.debug(f"generating {cmakelists_path}")
    cmakelist = generate_cmakelist_with_spatz(test, operators, format, arch)
    with open(cmakelists_path, "w") as f:
        f.write(cmakelist)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-t', '--test', type=str, required=True)
    parser.add_argument('-v', '--verbose', action='count', default=0)

    args = parser.parse_args()

    # logger configuration
    if args.verbose == 0:
        log_level = logging.WARNING
        coloredlogs.install(level='WARNING', logger=DEFAULT_LOGGER, fmt=DEFAULT_FMT)
    elif args.verbose == 1:
        log_level = logging.INFO
        coloredlogs.install(level='INFO', logger=DEFAULT_LOGGER, fmt=DEFAULT_FMT)
    else:
        log_level = logging.DEBUG
        coloredlogs.install(level='DEBUG', logger=DEFAULT_LOGGER, fmt=DEFAULT_FMT)

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.debug(f"args: {args}")
    main(args.test)
