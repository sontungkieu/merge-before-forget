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

## Llama-2-7B-chat reproduction gate

The paper-cell Llama track was added at commit
`7ddbdcaa70fa1f9360683efa3d69fc2f98969c7b` with the two published SuperNI
orders, paper-reported optimization settings, and pinned canonical checkpoint
`meta-llama/Llama-2-7b-chat-hf` revision
`f5db02db724555f92da89c216ac04704f23d4590`.

Two seed-42 O1 canaries reached the real model-load gate on one A100 3g.40gb
MIG slice:

| Job | Source | Verified outcome |
|---:|---|---|
| `45648515` | `7ddbdca` | Ruff, 20 tests, and development gates passed; gated checkpoint request returned HTTP 403 before model initialization. |
| `45648524` | `8471d85` | Credential-path integration fix was active, but the canonical checkpoint again returned `GatedRepoError`/HTTP 403 before model initialization. |

An independent login-node `hf_hub_download` check using the same ambient token
path also returned `GatedRepoError`. The credential was present and readable,
but its Hugging Face account was not authorized for the canonical Meta
repository. No training started, neither job is a scientific result, and no
third-party checkpoint mirror was substituted.

The blocker was resolved without changing the model or scientific
configuration at commit
`6548c74f9e472087a45f9058ba4f2c9627b00bde`. Following
`operate-talapas` v0.1.1, the runner now requires an explicitly selected stable
credential profile, clears ambient Hugging Face credentials, verifies the
expected non-secret account identity, and downloads `config.json` from the
exact pinned revision before training. The selected profile resolved to
account `codemaivanngu`; both login-node and compute-node probes passed.

Canary job `45648616` completed normally on one A100 3g.40gb MIG slice. It
passed Ruff, 22 tests, development gates, loaded the canonical model in
`torch.bfloat16`, trained/evaluated one real SuperNI task, and produced a
50,446,046-byte adapter checkpoint with SHA-256
`908c757e856469204827cbf1dc369edd001bd0cd1c6cd0be1c0fb2e342c1abf6`.
Its run-directory audit found zero sensitive artifacts. This is a development
gate, not a paper result.

Matched full O1 seed-42 SLAO and SeqLoRA jobs `45648618` and `45648619` were
then submitted at the same source commit and both passed the compute-node
identity, canonical-revision, A100 40 GB, and BF16 model-ready gates. Their
terminal 15-task artifacts remain pending and must be audited before either
run is reported as a result.

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

## Verified seed-44 runs

Jobs `45603674`, `45603675`, and `45603677` completed normally on one A100
MIG slice each at source commit
`23b2241c9eb611ab5a8b58c87a992ea8a895c62a`. Each compact local bundle passes
its SHA-256 manifest, contains 15 contiguous metric rows and a terminal
`status=completed` summary, records an empty Git status and
`torch.bfloat16`, and excludes checkpoints and prediction dumps.

| Method | Job | AA (%) | BWT (pp) | Runtime (s) |
|---|---:|---:|---:|---:|
| SLAO | `45603674` | 50.1809 | -3.2200 | 8,421.11 |
| SeqLoRA | `45603675` | 40.7451 | -16.9546 | 8,237.00 |
| `slao_merged_b_init` | `45603677` | 49.3416 | -4.0258 | 8,113.35 |

The compact evidence is in
`evidence/talapas/full-slao-s44-45603674/`,
`evidence/talapas/full-seqlora-s44-45603675/`, and
`evidence/talapas/full-merged-b-init-s44-45603677/`.

The intended seed-42 merged-B variant, job `45603676`, failed before model
initialization when its pinned SAPT `git fetch` returned exit 128. Its stderr
and `sacct` record remain in `evidence/talapas/infra-failures/`; it is not a
scientific result and was not silently retried. Consequently the third-party
variant has two verified seeds, not a three-seed aggregate.

## Three-seed primary aggregate

The pre-registered primary seeds 42, 43, and 44 are terminal and
artifact-verified for both SLAO and SeqLoRA.

| Method | AA mean ± sample SD (%) | BWT mean ± sample SD (pp) | Runtime mean ± sample SD (s) |
|---|---:|---:|---:|
| SLAO | 51.0375 ± 1.3629 | -2.2638 ± 1.7875 | 8,605.84 ± 305.17 |
| SeqLoRA | 43.6469 ± 2.6373 | -13.1557 ± 3.4462 | 8,253.16 ± 23.89 |

Across these three seeds, SLAO is +7.3906 AA points above SeqLoRA and has
10.8919 points less-negative BWT. The SLAO mean is +13.2375 points, or 35.02%
target-relative, above the paper's 37.8% row and remains outside the registered
5% tolerance. The execution is therefore a three-seed result for this
disclosed setup, but still a **partial numerical non-reproduction** of the
paper value.

The source commits differ across the seed batches. An explicit diff audit found
that the seed-43 delta adds resume support, tests, and the separately labeled
merged-B variant; the only primary config change is explanatory text. The
seed-44 delta only serializes the shared SAPT fetch. Neither changes the fresh
SLAO or SeqLoRA scientific path. The machine-readable aggregate and source
audit are in `evidence/talapas/primary-three-seed-20260723/summary.json`.
