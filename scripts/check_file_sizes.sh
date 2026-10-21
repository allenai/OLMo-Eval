#!/usr/bin/env bash
# Fail if any tracked file is larger than MAX_BYTES (default 1 MB).
#
# Every tracked file is checked, not only the ones a pull request touches.
# Pull requests are squash-merged, so the tree is what reaches main, and a
# file that is added and then removed on a branch never lands.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 1

MAX_BYTES=${MAX_BYTES:-$((1024 * 1024))}

# Paths allowed to exceed the limit. Keep this list short.
EXCLUDE=(
    uv.lock
)

human() {
    awk -v b="$1" 'BEGIN {
        split("B KB MB GB", u, " "); i = 1
        while (b >= 1024 && i < 4) { b /= 1024; i++ }
        printf "%.2f %s", b, u[i]
    }'
}

is_excluded() {
    local path="$1" excluded
    for excluded in "${EXCLUDE[@]}"; do
        [[ "$path" == "$excluded" ]] && return 0
    done
    return 1
}

offenders=()
count=0
# Sizes are of the blobs in the index, which is what gets committed.
# NUL-separated so that paths with spaces or newlines are safe.
while IFS= read -r -d '' entry; do
    count=$((count + 1))
    size=${entry%% *}
    path=${entry#* }
    is_excluded "$path" && continue
    # Submodules have no blob and count as zero.
    [[ "$size" =~ ^[0-9]+$ ]] || size=0
    if ((size > MAX_BYTES)); then
        offenders+=("$(printf '  %10s  %s' "$(human "$size")" "$path")")
    fi
done < <(git ls-files -z --format='%(objectsize) %(path)')

if ((${#offenders[@]} > 0)); then
    echo "These tracked files exceed $(human "$MAX_BYTES"):" >&2
    printf '%s\n' "${offenders[@]}" >&2
    echo >&2
    echo "Remove them from the change, add them to .gitignore, or store them outside the repo." >&2
    exit 1
fi

echo "All $count tracked files are within $(human "$MAX_BYTES")."
