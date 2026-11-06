# Copyright (C) 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
#
# Authors: Alberto Dequino <lasagne.arrosto@gmail.com>

# Environment setup script
# Sets up the shell environment to enable test compilation and running outside of the sdk folder.
# 
#   Usage:
#           source setup_env.sh <MAGIA_RTL_PATH>
# TODO: INSERT HERE ALL OTHER STUFF TO BE TAKEN INTO CONSIDERATION BY THE USER or something

export MAGIA_SDK_HOME=$(dirname $(dirname "$(readlink -f "${BASH_SOURCE[0]}")"))
export MAGIA_SDK=$MAGIA_SDK_HOME
export MAGIA_RTL_DIR=${1:-$(dirname $(dirname $(dirname "$(readlink -f "${BASH_SOURCE[0]}")")))}

