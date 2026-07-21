# Talapas evidence report

## Environment and execution contract

- User/account: `tnguye11` / Slurm account `ailab`.
- Access path: existing OpenSSH path `cmorl-gcp` to `talapas`, validated with a
  read-only preflight; no SSH configuration was changed.
- Project checkout:
  `/gpfs/projects/ailab/tnguye11/sontungkieu/merge-before-forget`.
- Evidence root:
  `/gpfs/projects/ailab/tnguye11/sontungkieu/merge-before-forget-evidence`.
- Jobs request one A100 3g.40gb MIG slice, use `--no-requeue`, unique labels,
  isolated output directories, and frozen `uv` dependencies. All repository
  Python entrypoints are invoked through `uv run`.

Slurm terminal state and `training/summary.json` must both validate before a
job is treated as a scientific result.

## Smoke gate

The first allocation, job `45580901`, remained pending for preemption and was
canceled before it started (zero elapsed compute). The transparent
infrastructure retry, job `45581455`, completed on node `n0151` using one
NVIDIA A100 80GB PCIe MIG 3g.40gb slice. It ran commit
`f137983960487b501ebedc4eff83257e20ad5c5f` and passed Ruff, 13 unit tests,
one-task overfit, three-task synthetic sequence, serialization, and a tiny
Qwen2.5-3B canary. The tiny 8-train/4-test, one-step Samsum smoke scored
25.9655 Rouge-L in 5.93 seconds; it is development evidence and not a paper
cell. Downloaded evidence is in `evidence/talapas/smoke-45581455/`.

## Full matched runs

Both jobs use commit `f137983960487b501ebedc4eff83257e20ad5c5f`, the same
pinned Qwen2.5-3B checkpoint revision, SuperNI O1 task sequence, seed 42,
optimizer, LoRA, and evaluation configuration. They differ only in continual
method.

| Method | Slurm job | Node | Run label | Current result state |
|---|---:|---|---|---|
| SLAO | `45581529` | `n0151` | `talapas-paper-slao-qwen3b-superni-o1-s42-20260722` | running; no result claimed |
| SeqLoRA | `45581530` | `n0156` | `talapas-baseline-seqlora-qwen3b-superni-o1-s42-20260722` | running; no result claimed |

Final scheduler state, elapsed compute, downloaded artifact paths, AA/BWT,
paper deltas, and scientific classification are added only after both jobs are
terminal and their structured outputs have been independently verified.
