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

## Full matched seed-42 runs

Both jobs use commit `f137983960487b501ebedc4eff83257e20ad5c5f`, the same
pinned Qwen2.5-3B checkpoint revision, SuperNI O1 task sequence, seed 42,
optimizer, LoRA, and evaluation configuration. They differ only in continual
method.

| Method | Slurm job | Node | Run label | Current result state |
|---|---:|---|---|---|
| SLAO | `45581529` | `n0151` | `talapas-paper-slao-qwen3b-superni-o1-s42-20260722` | `COMPLETED`, exit `0:0`; artifact-verified partial result |
| SeqLoRA | `45581530` | `n0156` | `talapas-baseline-seqlora-qwen3b-superni-o1-s42-20260722` | `COMPLETED`, exit `0:0`; artifact-verified matched baseline |
| FTBA-MB-style | `45586833` | `n0169` | `talapas-baseline-ftba-mb-qwen3b-superni-o1-s42-20260722` | `COMPLETED`, exit `0:0`; artifact-verified merging control |

## Verified primary results

| Method | AA (%) | BWT (pp) | Training/evaluation runtime | Slurm elapsed |
|---|---:|---:|---:|---:|
| SLAO | 50.3226 | -3.3698 | 8,958.08 s | 02:29:37 |
| SeqLoRA | 45.8978 | -10.2305 | 8,241.89 s | 02:18:47 |
| FTBA-MB-style | 50.3737 | -3.2420 | 8,178.89 s | 02:17:52 |

SLAO is +4.4248 AA points above matched SeqLoRA and its BWT is +6.8606
points less negative. Against the paper's Qwen2.5-3B SuperNI O1 SLAO target of
37.8%, the observed SLAO AA is +12.5226 points (33.13% target-relative, or
24.88% under the pre-registered symmetric relative delta). It is outside the
5% stochastic tolerance. This therefore demonstrates successful execution and
a matched advantage over SeqLoRA under the disclosed setup, but does not
reproduce the paper's numerical table value.

The FTBA-MB-style merging control is +0.0511 AA points and +0.1278 BWT points
above SLAO. Those tiny one-seed differences are not evidence of superiority,
but they do mean this run cannot support a claim that SLAO outperforms the
relevant merging control. This implementation is deliberately labeled
"FTBA-MB-style": it retains the same asymmetric B merge but initializes the
next task from the previous fine-tuned LoRA without SLAO's QR-normalized A.

Both outcomes are **partial**: they use one pre-registered seed, assumptions for
paper-omitted Qwen settings, and every unique row in the paper-cited SAPT
splits rather than the paper's impossible 1,000/100/100 claim. MOPD/AOPD are
not defined for a single task order.

The independently downloaded, hash-checked evidence is in
`evidence/talapas/full-slao-45581529/` and
`evidence/talapas/full-seqlora-45581530/`, and
`evidence/talapas/full-ftba-mb-45586833/`. Each directory contains `sacct`,
hardware, git revision/status, manifest, all 15 metric rows, score matrix,
summary, and logs. Adapter checkpoints and prediction dumps remain on GPFS and
were intentionally not downloaded.

The first FTBA-MB submission, `45586713`, was canceled while still pending at
zero elapsed time because it inherited the script's `preempt` partition. Job
`45586833` preserves the experiment identity and uses the same `gpu/normal`
profile as the verified primary jobs; this is an infrastructure correction,
not a scientific retry.

## Verified seed-43 runs

Jobs `45603663`, `45603665`, and `45603668` completed normally on one A100
MIG slice each at source commit
`b47cc8f0fec63e520c4ca5646a84f39b08109f70`. Each run has exactly 15 metric
rows, a terminal `status=completed` summary, an empty Git status, and a
`torch.bfloat16` model-ready event.

| Method | Job | AA (%) | BWT (pp) | Runtime (s) |
|---|---:|---:|---:|---:|
| SLAO | `45603663` | 52.6092 | -0.2016 | 8,438.32 |
| SeqLoRA | `45603665` | 44.2980 | -12.2819 | 8,280.61 |
| `slao_merged_b_init` | `45603668` | 51.1486 | -3.0806 | 8,188.47 |

For seed 43, SLAO is +8.3112 AA points above SeqLoRA and has 12.0803 points
less-negative BWT. The `slao_merged_b_init` run is an audited comparison to
the available third-party implementation, but it is **not faithful to
Algorithm 1**: it initializes the next task's B factor from the merged state,
whereas the paper specifies the previous fine-tuned B factor. It is therefore
reported as a variant, not as an independent reproduction of paper SLAO.

Compact, hash-checked evidence is in
`evidence/talapas/full-slao-s43-45603663/`,
`evidence/talapas/full-seqlora-s43-45603665/`, and
`evidence/talapas/full-merged-b-init-s43-45603668/`. Adapter checkpoints and
prediction dumps remain on GPFS.

The seed-44 SLAO, SeqLoRA, and merged-B variant jobs are running as
`45603674`, `45603675`, and `45603677`. The intended seed-42 merged-B variant,
job `45603676`, failed before model initialization when its pinned SAPT
`git fetch` returned exit 128. Its stderr and `sacct` record are preserved in
`evidence/talapas/infra-failures/`; it is not a scientific result. The fetcher
now reuses an already verified commit with all 45 split files instead of
unconditionally contacting the remote. No three-seed mean is reported until
the remaining terminal runs pass the same artifact gates.
