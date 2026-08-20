#!/usr/bin/env bash
set -euo pipefail

sdk_root="$(cd "$(dirname "$0")/../.." && pwd)"
binary="${TMPDIR:-/tmp}/maps-operation-indexing-test"
"${CC:-cc}" -std=c11 -Wall -Wextra -Werror \
    -I"${sdk_root}/targets/magia_v3/include" \
    "${sdk_root}/tests/magia/maps_operation_indexing.c" -o "${binary}"
"${binary}"
