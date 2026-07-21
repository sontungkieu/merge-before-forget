#!/usr/bin/env bash

set -euo pipefail

EXPECTED_SHA="${1:?usage: bootstrap_remote.sh EXPECTED_SHA}"
REMOTE_PROJECT="${SLAO_REMOTE_PROJECT:-/gpfs/projects/ailab/tnguye11/sontungkieu/merge-before-forget}"
EVIDENCE_ROOT="${SLAO_EVIDENCE_ROOT:-/gpfs/projects/ailab/tnguye11/sontungkieu/merge-before-forget-evidence}"
SCRATCH_ROOT="${SLAO_SCRATCH_ROOT:-/scratch/ailab/tnguye11/slao-repro}"
UV_VERSION="0.10.2"
UV_BIN="${SCRATCH_ROOT}/tools/uv"

mkdir -p "$(dirname "${REMOTE_PROJECT}")" "${EVIDENCE_ROOT}/slurm" "${SCRATCH_ROOT}/tools"

if [[ ! -d "${REMOTE_PROJECT}/.git" ]]; then
  git clone --branch repro/talapas \
    https://github.com/sontungkieu/merge-before-forget.git "${REMOTE_PROJECT}"
fi
git -C "${REMOTE_PROJECT}" fetch --prune origin repro/talapas
git -C "${REMOTE_PROJECT}" checkout --detach "${EXPECTED_SHA}"
ACTUAL_SHA="$(git -C "${REMOTE_PROJECT}" rev-parse HEAD)"
if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
  printf 'checkout mismatch: expected %s, got %s\n' "${EXPECTED_SHA}" "${ACTUAL_SHA}" >&2
  exit 1
fi
if [[ -n "$(git -C "${REMOTE_PROJECT}" status --porcelain --untracked-files=no)" ]]; then
  printf 'tracked remote checkout is dirty; refusing bootstrap\n' >&2
  exit 1
fi

if [[ ! -x "${UV_BIN}" ]] || [[ "$("${UV_BIN}" --version)" != "uv ${UV_VERSION}" ]]; then
  ARCHIVE="${SCRATCH_ROOT}/tools/uv-${UV_VERSION}.tar.gz"
  curl --fail --location --retry 3 \
    "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
    --output "${ARCHIVE}"
  tar -xzf "${ARCHIVE}" -C "${SCRATCH_ROOT}/tools"
  install -m 0755 "${SCRATCH_ROOT}/tools/uv-x86_64-unknown-linux-gnu/uv" "${UV_BIN}"
fi

export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/venv"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache"
"${UV_BIN}" sync --project "${REMOTE_PROJECT}" --frozen --extra test
printf 'BOOTSTRAP_OK sha=%s uv=%s environment=%s\n' \
  "${ACTUAL_SHA}" "$("${UV_BIN}" --version)" "${UV_PROJECT_ENVIRONMENT}"
