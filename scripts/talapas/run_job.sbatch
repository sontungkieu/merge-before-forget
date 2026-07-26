#!/usr/bin/env bash
#SBATCH --account=ailab
#SBATCH --partition=preempt
#SBATCH --constraint=a100
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --no-requeue

set -euo pipefail

RUN_KIND="${RUN_KIND:?RUN_KIND must be smoke or paper}"
RUN_LABEL="${RUN_LABEL:?RUN_LABEL is required}"
EXPECTED_SHA="${EXPECTED_SHA:?EXPECTED_SHA is required}"
METHOD="${METHOD:-slao}"
SEED="${SEED:-42}"
CONFIG_PATH="${CONFIG_PATH:-configs/paper/qwen25_3b_superni_o1.yaml}"
REMOTE_PROJECT="${SLAO_REMOTE_PROJECT:-/gpfs/projects/ailab/tnguye11/sontungkieu/merge-before-forget}"
EVIDENCE_ROOT="${SLAO_EVIDENCE_ROOT:-/gpfs/projects/ailab/tnguye11/sontungkieu/merge-before-forget-evidence}"
SCRATCH_ROOT="${SLAO_SCRATCH_ROOT:-/scratch/ailab/tnguye11/slao-repro}"
UV_BIN="${SCRATCH_ROOT}/tools/uv"
RUN_DIR="${EVIDENCE_ROOT}/runs/${RUN_LABEL}"

if [[ ! "${RUN_LABEL}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'invalid run label: %s\n' "${RUN_LABEL}" >&2
  exit 64
fi
if [[ ! "${RUN_KIND}" =~ ^(smoke|paper)$ ]]; then
  printf 'invalid run kind: %s\n' "${RUN_KIND}" >&2
  exit 64
fi
if [[ ! "${METHOD}" =~ ^(slao|slao_merged_b_init|seqlora|ftba_mb)$ ]]; then
  printf 'invalid method: %s\n' "${METHOD}" >&2
  exit 64
fi
if [[ ! "${CONFIG_PATH}" =~ ^configs/paper/[A-Za-z0-9._-]+\.yaml$ ]]; then
  printf 'invalid config path: %s\n' "${CONFIG_PATH}" >&2
  exit 64
fi
if [[ ! -f "${REMOTE_PROJECT}/${CONFIG_PATH}" ]]; then
  printf 'missing config: %s\n' "${REMOTE_PROJECT}/${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -x "${UV_BIN}" ]]; then
  printf 'missing pinned uv binary: %s\n' "${UV_BIN}" >&2
  exit 1
fi
ACTUAL_SHA="$(git -C "${REMOTE_PROJECT}" rev-parse HEAD)"
if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
  printf 'checkout mismatch: expected %s, got %s\n' "${EXPECTED_SHA}" "${ACTUAL_SHA}" >&2
  exit 1
fi
if [[ -d "${RUN_DIR}" ]] && [[ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  printf 'refusing non-empty run directory: %s\n' "${RUN_DIR}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}" "${SCRATCH_ROOT}/hf" "${SCRATCH_ROOT}/uv-cache" "${SCRATCH_ROOT}/locks"
export UV_PROJECT_ENVIRONMENT="${SCRATCH_ROOT}/venv"
export UV_CACHE_DIR="${SCRATCH_ROOT}/uv-cache"
export HF_HOME="${SCRATCH_ROOT}/hf"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd "${REMOTE_PROJECT}"

printf '%s\n' "${ACTUAL_SHA}" > "${RUN_DIR}/git_revision.txt"
git status --short > "${RUN_DIR}/git_status.txt"
"${UV_BIN}" --version > "${RUN_DIR}/uv_version.txt"
nvidia-smi > "${RUN_DIR}/nvidia_smi.txt"
nvidia-smi \
  --query-gpu=timestamp,name,uuid,memory.total,driver_version,compute_cap \
  --format=csv,noheader > "${RUN_DIR}/hardware.csv"
lscpu > "${RUN_DIR}/lscpu.txt"
scontrol show job "${SLURM_JOB_ID}" > "${RUN_DIR}/slurm_job.txt"

AUTHENTICATION="$(
  "${UV_BIN}" run --no-sync python -c \
    'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["model"].get("authentication", "none"))' \
    "${CONFIG_PATH}"
)"
case "${AUTHENTICATION}" in
  huggingface_token)
    # shellcheck source=scripts/cluster/load_talapas_env.sh
    source scripts/cluster/load_talapas_env.sh
    "${UV_BIN}" run --no-sync python -m slao_repro.hf_preflight \
      --config "${CONFIG_PATH}" | tee "${RUN_DIR}/hf_identity_preflight.log"
    ;;
  none)
    ;;
  *)
    printf 'unsupported model authentication mode: %s\n' "${AUTHENTICATION}" >&2
    exit 64
    ;;
esac

flock --wait 900 "${SCRATCH_ROOT}/locks/sapt-fetch.lock" \
  "${UV_BIN}" run --no-sync python scripts/fetch_benchmark_data.py \
  --benchmark superni --destination data/SAPT | tee "${RUN_DIR}/data_fetch.log"

if [[ "${RUN_KIND}" == "smoke" ]]; then
  "${UV_BIN}" run --no-sync ruff check . | tee "${RUN_DIR}/ruff.log"
  "${UV_BIN}" run --no-sync pytest -q -s | tee "${RUN_DIR}/pytest.log"
  "${UV_BIN}" run --no-sync python scripts/dev_gates.py \
    --output-dir "${RUN_DIR}/development" | tee "${RUN_DIR}/development.log"
  "${UV_BIN}" run --no-sync python -m slao_repro.train \
    --config "${CONFIG_PATH}" \
    --method "${METHOD}" --seed "${SEED}" --run-label "${RUN_LABEL}" \
    --output-dir "${RUN_DIR}/training" --max-tasks 1 --max-train-samples 8 \
    --max-eval-samples 4 --epochs 1 --max-optimizer-steps-per-task 2 \
    | tee "${RUN_DIR}/training.log"
else
  "${UV_BIN}" run --no-sync python -m slao_repro.train \
    --config "${CONFIG_PATH}" \
    --method "${METHOD}" --seed "${SEED}" --run-label "${RUN_LABEL}" \
    --output-dir "${RUN_DIR}/training" | tee "${RUN_DIR}/training.log"
fi

printf 'TALAPAS_RUN_COMPLETE job_id=%s run_label=%s kind=%s method=%s\n' \
  "${SLURM_JOB_ID}" "${RUN_LABEL}" "${RUN_KIND}" "${METHOD}" \
  | tee "${RUN_DIR}/completion.txt"
