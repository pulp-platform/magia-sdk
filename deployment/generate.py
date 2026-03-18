
import os
import shutil
import re
from pathlib import Path
from typing import Optional, Sequence
from argparse import ArgumentParser
import logging

import numpy as np
from numpy.typing import NDArray

import onnx
import onnx_graphsurgeon as gs

import Deeploy
from Deeploy.AbstractDataTypes import Pointer, PointerClass
from Deeploy.CommonExtensions.NetworkDeployers.SignPropDeployer import SignPropDeployer
from Deeploy.CommonExtensions import DataTypes
from Deeploy.CommonExtensions.DataTypes import FloatDataTypes, IntegerDataTypes

from MagiaDeeployTarget.Deployer import MagiaDeployer
from MagiaDeeployTarget.Platform import MagiaPlatform, MagiaOptimizer


copyright_str = """
Copyright 2026 Fondazione ChipsIT.
Licensed under the Apache License, Version 2.0, see LICENSE for details.
SPDX-License-Identifier: Apache-2.0

Alex Marchioni <alex.marchioni@chips.it>
"""


def copyright_comment(comment_symbol:str = "//") -> str:
    text = ''.join([f'{comment_symbol} {row}\n'
                    for row in copyright_str.split("\n")[1:-1]])
    return text


def defaultScheduler(graph: gs.Graph):
    return graph.nodes


def load_npz(path: Path) -> list[NDArray]:
    npz = np.load(path)
    arrays = [npz[key] for key in npz.files]
    return arrays


def generate_array_size(array: NDArray, name: Optional[str] = None) -> str:
    if name is None:
        name = "array"
    name = name.upper()  # ensure upper case
    _str = f"#define {name}_NDIM {array.ndim}\n"
    for i, dim in enumerate(array.shape):
        _str += f"#define {name}_DIM{i} {dim}\n"
    _str += f"#define {name}_SIZE {np.prod(array.shape)}\n"
    _str += f"#define {name}_TYPE {array.dtype}_t\n"
    return _str

def generate_arrays_size(
        arrays: Sequence[NDArray],
        name: Optional[str] = None,
        ) -> str:
    if name is None:
        name = "array"
    _str = f"#define {name.upper()}S_NUM {len(arrays)}\n\n"
    _str += "".join([
        generate_array_size(array, f"{name}{i}") + "\n"
        for i, array in enumerate(arrays)
    ])
    return _str


def generate_array(array: NDArray, name: Optional[str] = None) -> str:
    if name is None:
        name = "array"
    array_flat = array.reshape(-1)  # reshape as a 1D array
    _str = f"{array.dtype}_t {name}[{name.upper()}_SIZE] = " + "{"
    if np.issubdtype(array.dtype, np.integer):  # integers
        _str += ", ".join([str(x) for x in array_flat])
    elif np.issubdtype(array.dtype, np.float):  # floats
        _str += ", ".join(
            [f'{x}f' if np.isfinite(x) else str(x) for x in array_flat])
    else:
        raise ValueError(f"array type {array.dtype} not supported.")
    _str += "};\n"
    return _str


def generate_arrays(arrays: list[NDArray], name: Optional[str] = None) -> str:
    if name is None:
        name = "array"
    names = [f"{name}{i}" for i in range(len(arrays))]

    # generate each array
    _str = "".join([
        generate_array(array.reshape(-1), _name) + "\n"
        for _name, array in zip(names, arrays)
    ])

    # generate the array containing all arrays
    _str += f"void* {name}s[{name.upper()}S_NUM] = {{{', '.join(names)}}};\n"

    # generate the array containing the size of each array
    size_list = [f"{_name.upper()}_SIZE" for _name in names]
    _str += f"uint32_t {name}s_size[{name.upper()}S_NUM] = "\
            f"{{{', '.join(size_list)}}};\n"

    # generate the array containing the  element size of each array
    elem_size_list = [f"sizeof({_name.upper()}_TYPE)" for _name in names]
    _str += f"uint32_t {name}s_elem_size[{name.upper()}S_NUM] = "\
            f"{{{', '.join(elem_size_list)}}};\n"
    return _str


def generate_test_header(inputs: list[NDArray], outputs: list[NDArray]) -> str:

    guard_str = f"_TEST_INCLUDE_GUARD_"

    text = copyright_comment('//')
    text += "\n"
    text += f"#ifndef {guard_str}\n"
    text += f"#define {guard_str}\n"
    text += "\n"
    text += "#include <stdint.h>\n"
    text += "\n"
    text += generate_arrays_size(inputs, "input")
    text += "\n"
    text += generate_arrays_size(outputs, "output")
    text += "\n"
    text += generate_arrays(inputs, "input")
    text += "\n"
    text += generate_arrays(outputs, "output")
    text += "\n"
    text += f"#endif // {guard_str}\n"
    return text

def generate_network_header(deployer: MagiaDeployer) -> str:

    guard_str = f"_NETWORK_INCLUDE_GUARD_"

    text = copyright_comment('//')
    text += "\n"
    text += f"#ifndef {guard_str}\n"
    text += f"#define {guard_str}\n"
    text += "\n"
    text += "#include <stdint.h>\n"
    text += "\n"
    text += "void RunNetwork();\n"
    text += "void InitNetwork();\n"
    text += "\n"
    text += deployer.generateIOBufferInitializationCode() + "\n"
    text += "\n"
    text += f"#endif // {guard_str}\n"
    return text

def generate_network_source(deployer: MagiaDeployer) -> str:

    text = copyright_comment('//')
    text += "\n"
    text += deployer.generateIncludeString() +"\n"
    text += '#include "network.h"\n'
    text += "\n"
    text += deployer.generateBufferInitializationCode()
    text += "\n"
    text += deployer.generateGlobalDefinitionCode()
    text += "\n"
    text += "void RunNetwork() {\n"
    text += deployer.generateInferenceInitializationCode() + "\n"
    text += deployer.generateFunction() + "\n"
    text += "}\n"
    text += "\n"
    text += "void InitNetwork() {\n"
    text += deployer.generateEngineInitializationCode()
    text += deployer.generateBufferAllocationCode()
    text += "}\n"

    return text

def generate_cmakelist(test: str) -> str:

    text = copyright_comment('#')
    text += "\n"
    text += f"set(TEST_NAME test_{test})\n"
    text += "\n"
    text += 'file(GLOB_RECURSE TEST_SRCS "src/*.c")\n'
    text += "\n"
    text += "add_executable(${TEST_NAME} ${TEST_SRCS})\n"
    text += "target_include_directories(${TEST_NAME} PUBLIC include)\n"
    text += "\n"
    text += "target_compile_options(${TEST_NAME} PRIVATE -O2)\n"
    text += "target_link_libraries(${TEST_NAME} PUBLIC runtime hal kernels)\n"
    text += "\n"
    text += "add_custom_command(\n"
    text += "    TARGET ${TEST_NAME}\n"
    text += "    POST_BUILD\n"
    text += "    COMMAND ${CMAKE_OBJDUMP} -dhS -Mmarch=${ISA} "
    text += " $<TARGET_FILE:${TEST_NAME}> > $<TARGET_FILE:${TEST_NAME}>.s\n"
    text += ")\n"
    return text


def allocator_patch(
        code: str,
        inputs: Sequence[NDArray],
        outputs: Sequence[NDArray],
    ) -> str:
    for i, _input in enumerate(inputs):
        code = code.replace(
            f'*DeeployNetwork_input_{i};',
            f'DeeployNetwork_input_{i}[{_input.size}];',
        )
        code = code.replace(
            f'* DeeployNetwork_input_{i};',
            f' DeeployNetwork_input_{i}[{_input.size}];',
        )
        code = code.replace(
            f'DeeployNetwork_input_{i} =',
            f'// DeeployNetwork_input_{i} =',
        )
    for i, _output in enumerate(outputs):
        code = code.replace(
            f'*DeeployNetwork_output_{i};',
            f'DeeployNetwork_output_{i}[{_output.size}];',
        )
        code = code.replace(
            f'* DeeployNetwork_output_{i};',
            f' DeeployNetwork_output_{i}[{_output.size}];',
        )
        code = code.replace(
            f'DeeployNetwork_output_{i} =',
            f'// DeeployNetwork_output_{i} =',
        )
    return code

def main(src_dir: Path, dst_dir: Path) -> None:
    """
    src_dir
    ├── inputs.npz
    ├── outputs.npz
    └── network.onnx

    dst_dir
    ├── include
    |   ├── network.h
    │   └── test.h
    ├── src
    |   ├── network.c
    │   └── test.c
    └── CMakeLists.txt
    """

    # load inputs, outputs, and network
    logger.debug("loading inputs and outputs data")
    inputs = load_npz(src_dir / 'inputs.npz')
    outputs = load_npz(src_dir / 'outputs.npz')

    logger.debug("loading onnx model")
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
        default_channels_first=False,
        deeployStateDir="states",
    )

    # run deployment process to be ready to generate code
    logger.debug("run deployment process")
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

    # header for test inputs and outputs
    test_header_path = dst_inc_dir / 'test.h'
    logger.debug(f"generate {test_header_path}")
    test_header = generate_test_header(inputs, outputs)
    with open(test_header_path, "w") as f:
        f.write(test_header)
    os.system(clang_cmd(test_header_path))

    # header for network
    network_header_path = dst_inc_dir / 'network.h'
    logger.debug(f"generate {network_header_path}")
    network_header = generate_network_header(deployer)
    network_header = allocator_patch(network_header, inputs, outputs) # patch
    with open(network_header_path, "w") as f:
        f.write(network_header)
    os.system(clang_cmd(network_header_path))


    # source for network
    network_source_path = dst_src_dir / 'network.c'
    logger.debug(f"generate {network_source_path}")
    network_source = generate_network_source(deployer)
    network_source = allocator_patch(network_source, inputs, outputs) # patch
    with open(network_source_path, "w") as f:
        f.write(network_source)
    os.system(clang_cmd(network_source_path))

    # test main
    test_path = dst_src_dir / 'test.c'
    logger.debug(f"generate {test_path}")
    shutil.copyfile(src_dir.parents[1] / 'test.c', test_path)

    # CMakeLists
    cmakelists_path = dst_dir / 'CMakeLists.txt'
    logger.debug(f"generate {cmakelists_path}")
    cmakelist = generate_cmakelist('adder')
    with open(cmakelists_path, "w") as f:
        f.write(cmakelist)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-s', '--source', type=str, required=True)
    parser.add_argument('-d', '--destination', type=str, required=True)
    parser.add_argument('-v', '--verbose', action='count', default=0)

    args = parser.parse_args()


    # logger
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # generate code
    logger.debug(f"args: {args}")
    main(Path(args.source), Path(args.destination))

    # test_name = 'adder'
    # src_dir = Path('/home/alex/magia-sdk/deployment/tests/') / test_name
    # dst_dir = Path('/home/alex/magia-sdk/tests/magia/mesh') / test_name
    # main(src_dir, dst_dir)