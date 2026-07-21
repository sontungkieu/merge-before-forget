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
- This Qwen checkpoint is public, so no external token is required. A private
  secrets file is neither sourced nor inspected by these jobs.

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
`EXPECTED_SHA`, `METHOD`, and `SEED`. The job refuses a non-empty output
directory. `RUN_KIND=smoke` executes static/unit/development gates plus a tiny
one-task Qwen integration run; `RUN_KIND=paper` executes the full 15-task
configuration. Every Python entrypoint in the job is launched through
`uv run --no-sync`.

The preemptible partition is invoked with `--no-requeue`: a preemption is a
recorded terminal failure, not an invisible scientific retry. Hardware, Git
revision, Slurm state, stdout/stderr, structured metrics, predictions, and the
adapter-only checkpoint remain in the unique evidence directory. Only small
diagnostics and metrics should be copied back to this repository.
