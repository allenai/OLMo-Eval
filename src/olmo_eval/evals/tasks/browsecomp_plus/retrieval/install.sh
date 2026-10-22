#!/usr/bin/env bash
set -euo pipefail

java_21_available() {
    java -version 2>&1 | grep -Eq 'version "21([.]|")'
}

if ! java_21_available; then
    if ! sudo apt update || ! sudo apt install -y openjdk-21-jdk || ! java_21_available; then
        echo 'Error: Java 21 installation did not succeed. You can try another installation method.' >&2
        echo 'For example, without sudo: conda install -c conda-forge openjdk=21' >&2
        echo 'Make sure Java 21 is on PATH, then rerun this script.' >&2
        exit 1
    fi
fi

retrieval_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
artifact_dir="${BROWSECOMP_PLUS_HOME:-$HOME/.cache/olmo-eval/browsecomp-plus}"
mkdir -p -- "$artifact_dir"
export UV_PROJECT_ENVIRONMENT="$(cd -- "$artifact_dir" && pwd)/retrieval-venv"

uv venv --no-project --python 3.10 --allow-existing "$UV_PROJECT_ENVIRONMENT"
uv sync --project "$retrieval_dir/upstream" --frozen
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-build-isolation flash-attn
