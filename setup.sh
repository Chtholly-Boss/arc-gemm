#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Generate .clangd with repo-specific includes.
sed "s|__REPO_ROOT__|${REPO_ROOT}|g" "${REPO_ROOT}/.clangd.template" > "${REPO_ROOT}/.clangd"

# Detect CUDA version from nvcc.
if command -v nvcc > /dev/null 2>&1; then
    NVCC_PATH="$(command -v nvcc)"
elif [ -x /usr/local/cuda/bin/nvcc ]; then
    NVCC_PATH="/usr/local/cuda/bin/nvcc"
else
    echo "error: nvcc not found in PATH or /usr/local/cuda/bin/nvcc" >&2
    exit 1
fi

NVCC_VER_LINE="$("${NVCC_PATH}" --version | grep "release" || true)"
if [[ "${NVCC_VER_LINE}" =~ release[[:space:]]+([0-9]+)\.([0-9]+) ]]; then
    CUDA_MAJOR="${BASH_REMATCH[1]}"
    CUDA_MINOR="${BASH_REMATCH[2]}"
else
    echo "error: failed to parse CUDA version from: ${NVCC_VER_LINE}" >&2
    exit 1
fi

# Detect GPU compute capability from nvidia-smi.
if ! command -v nvidia-smi > /dev/null 2>&1; then
    echo "error: nvidia-smi not found in PATH" >&2
    exit 1
fi

ARCH_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:].')"
if [[ -z "${ARCH_CAP}" || ! "${ARCH_CAP}" =~ ^[0-9]+$ ]]; then
    echo "error: failed to detect GPU compute capability from nvidia-smi" >&2
    exit 1
fi

CUDA_ARCH_SM="${ARCH_CAP}"
CUDA_ARCH="${ARCH_CAP}0"

echo "Using CUDA version: ${CUDA_MAJOR}.${CUDA_MINOR}"
echo "Using CUDA arch: sm_${CUDA_ARCH_SM} (__CUDA_ARCH__=${CUDA_ARCH})"

# Generate config.yaml from template.
sed -e "s|__CUDA_MAJOR__|${CUDA_MAJOR}|g" \
    -e "s|__CUDA_MINOR__|${CUDA_MINOR}|g" \
    -e "s|__CUDA_ARCH_SM__|${CUDA_ARCH_SM}|g" \
    -e "s|__CUDA_ARCH_VAL__|${CUDA_ARCH}|g" \
    "${REPO_ROOT}/config.yaml.template" > "${REPO_ROOT}/config.yaml"

mkdir -p "${HOME}/.config/clangd"
mv "${REPO_ROOT}/config.yaml" "${HOME}/.config/clangd/"
echo "Generated .clangd and config.yaml with REPO_ROOT=${REPO_ROOT}"
echo "Moved config.yaml to ${HOME}/.config/clangd/"
