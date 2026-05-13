#!/usr/bin/env bash
# Apply or check clang-format on C/C++ files changed on the current branch
# relative to its merge-base with main.
#
# Usage:
#   format-changed.sh apply   [--committed]
#   format-changed.sh check   [--committed]
#   format-changed.sh list    [--committed]
#
# By default (without --committed), the file set includes committed changes
# vs. merge-base(main) plus any staged or unstaged local edits, so `make format`
# operates on in-progress work. With --committed, only the merge-base diff is
# used (CI mode).

set -euo pipefail

MODE="${1:-}"
COMMITTED=0
shift || true
for arg in "$@"; do
    case "$arg" in
        --committed) COMMITTED=1 ;;
        *) echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

case "$MODE" in
    apply|check|list) ;;
    *) echo "Usage: $0 {apply|check|list} [--committed]" >&2; exit 2 ;;
esac

if ! command -v clang-format >/dev/null 2>&1; then
    echo "error: clang-format not found on PATH" >&2
    exit 2
fi

# Resolve merge-base with main. Prefer origin/main (CI), fall back to local main.
BASE=""
for ref in origin/main main; do
    if git rev-parse --verify --quiet "$ref" >/dev/null; then
        if BASE=$(git merge-base HEAD "$ref" 2>/dev/null); then
            break
        fi
    fi
done

if [ -z "$BASE" ]; then
    echo "error: could not find merge-base with main (need 'main' or 'origin/main')" >&2
    exit 2
fi

# Collect candidate files.
{
    git diff --name-only --diff-filter=ACMR "$BASE"...HEAD
    if [ "$COMMITTED" -eq 0 ]; then
        git diff --name-only --diff-filter=ACMR
        git diff --name-only --diff-filter=ACMR --cached
    fi
} | sort -u > /tmp/format-changed.$$.all

# Filter to C/C++ extensions and existing files only.
FILES=()
while IFS= read -r f; do
    case "$f" in
        *.c|*.h|*.cpp|*.hpp|*.cc|*.hh)
            [ -f "$f" ] && FILES+=("$f")
            ;;
    esac
done < /tmp/format-changed.$$.all
rm -f /tmp/format-changed.$$.all

if [ "${#FILES[@]}" -eq 0 ]; then
    case "$MODE" in
        list) ;;
        apply) echo "No changed C/C++ files to format." ;;
        check) echo "No changed C/C++ files to check." ;;
    esac
    exit 0
fi

case "$MODE" in
    list)
        printf '%s\n' "${FILES[@]}"
        ;;
    apply)
        echo "Formatting ${#FILES[@]} file(s)..."
        clang-format -i --style=file "${FILES[@]}"
        ;;
    check)
        echo "Checking format on ${#FILES[@]} file(s)..."
        clang-format --dry-run --Werror --style=file "${FILES[@]}"
        ;;
esac
