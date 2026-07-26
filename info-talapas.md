# Talapas runbook

This branch targets Talapas through the existing `cmorl-gcp -> talapas`
ControlMaster path. Connection setup is external to this repository; these
scripts do not edit SSH configuration.

## Verified read-only preflight (2026-07-22)

- Remote identity: `tnguye11@login1.talapas.uoregon.edu`.
- Project account: `ailab`; preempt QoS/partition is available.
- Paper-matched accelerator request: one 40 GB A100 MIG instance,
  `gpu:nvidia_a100_80gb_pcie_3g.40gb:1` with feature `a100`.
- Remote Python: 3.11.5; Git: 2.43.7; `uv` was not preinstalled.
- Project filesystem: `/gpfs/projects/ailab`; cache/scratch filesystem:
  `/scratch/ailab`.
- The Qwen checkpoint is public. Llama-2-7B-chat is gated. On the shared Unix
  account, never use `${HOME}/.cache/huggingface/token` or another ambient
  service cache. Select a stable absolute per-operator credential profile with
  `TALAPAS_SECRETS_ENV` and pass the non-secret expected identity through
  `SLAO_HF_EXPECTED_ACCOUNT`. `scripts/cluster/load_talapas_env.sh` rejects
  loose permissions and ambient credential overrides;
  `slao_repro.hf_preflight` verifies both the account identity and access to
  the exact pinned model revision before training. No token value is printed
  or persisted in evidence.

## Fixed paths

- Checkout: `/gpfs/projects/ailab/tnguye11/sontungkieu/merge-before-forget`
- Reviewed evidence: `/gpfs/projects/ailab/tnguye11/sontungkieu/merge-before-forget-evidence`
- Slurm logs: `${evidence}/slurm`
- Environment/cache: `/scratch/ailab/tnguye11/slao-repro`
- Pinned `uv`: `${scratch}/tools/uv`

## Execution contract

First run `scripts/talapas/bootstrap_remote.sh <expected-commit>` on the login
node. It checks out that exact public commit, installs the pinned standalone
`uv` binary, and performs `uv sync --frozen --extra test`. It never loads or
prints secrets.

Submit `scripts/talapas/run_job.sbatch` with explicit `RUN_KIND`, `RUN_LABEL`,
`EXPECTED_SHA`, `METHOD`, `SEED`, and, for non-Qwen runs, `CONFIG_PATH`. The
config path is restricted to a YAML file directly under `configs/paper/`. The
job refuses a non-empty output directory. `RUN_KIND=smoke` executes
static/unit/development gates plus a tiny one-task integration run;
`RUN_KIND=paper` executes the full 15-task configuration. Every Python
entrypoint in the job is launched through `uv run --no-sync`.

Before a gated-model submission, select and validate a stable credential
profile without reading its values:

```bash
export TALAPAS_SECRETS_ENV=/absolute/private/operator-project.env
"$SKILL_DIR/scripts/secrets_env.sh" check \
  --file "$TALAPAS_SECRETS_ENV" --require HF_TOKEN
```

Run the loader plus `python -m slao_repro.hf_preflight` in one scoped
subprocess before `sbatch`. Export only the stable `TALAPAS_SECRETS_ENV` path
and the expected non-secret account ID to Slurm; the allocated job repeats the
identity and exact-checkpoint access probe. Never overwrite the selected
profile while the job is queued or running.

The preemptible partition is invoked with `--no-requeue`: a preemption is a
recorded terminal failure, not an invisible scientific retry. Hardware, Git
revision, Slurm state, stdout/stderr, structured metrics, predictions, and the
adapter-only checkpoint remain in the unique evidence directory. Only small
diagnostics and metrics should be copied back to this repository.
