# Infrastructure comparison

The Talapas and Kaggle branches were created as independent worktrees from the
same base commit, `3e9bed0d1baeac907eaab21a526053400ac2c405`. Their paper-cell
runs use the same clean-room SLAO implementation, Qwen2.5-3B revision, pinned
SAPT data, O1 task order, seed 42, LoRA settings, optimizer, and metrics.

| Dimension | Talapas | Kaggle |
|---|---|---|
| Branch | `repro/talapas` | `repro/kaggle` |
| Full-run source commit | seeds use audited descendants summarized at `654f6fc` | final chunked pipelines use `b188dab065982c80e51af06f43ea7a31018f073e` |
| Scheduler/status evidence | Slurm `squeue`, `scontrol`, and `sacct` | KJO status history backed by Kaggle kernels API |
| GPU | A100 80GB PCIe 3g.40gb MIG slice | one Tesla P100-PCIE-16GB |
| Model dtype | BF16 | failed attempts used emulated BF16; corrected runner falls back to FP16 because P100 lacks native BF16 |
| Environment | Apptainer plus frozen `uv` project | Kaggle image plus frozen `uv` project |
| Durable raw evidence | GPFS evidence directory, then compact local pull | Kaggle outputs, then KJO diagnostics download |
| Secret mode | none for the public Qwen model | none inside notebook; local API credential only for KJO |

The commits differ only because each branch contains its infrastructure
workflow, and the Kaggle branch adds a pre-training model-shard canary after a
documented cache failure. Neither change alters Algorithm 1 or the scientific
configuration used by the full runs. Hardware and floating-point differences
can still cause stochastic drift, so cross-infrastructure agreement is
diagnostic rather than an exact determinism claim.

All Qwen O1 results remain **partial reproduction** even though execution
succeeds:
the paper omits Qwen-specific seeds, repository revision, LoRA alpha/dropout,
and complete decoding details; its claimed SuperNI sample cardinalities also
conflict with the cited pinned SAPT splits. Three pre-registered seeds establish
the reproduction mean for this disclosed setup, but cannot identify the
paper's undisclosed seeds. Tiny smoke runs remain **development** evidence, and
a terminal run lacking verified structured artifacts is **failed** or
**pending**, never successful.

The first Kaggle full attempts were cancelled after 9/15 tasks; corrected
native-FP16 retries reached 11/15 before the session boundary. Task-boundary
checkpointing then completed all three seeds without changing the scientific
configuration. The final cross-platform aggregates are:

| Platform | SLAO AA / BWT mean ± sample SD | SeqLoRA AA / BWT mean ± sample SD | Mean runtime, SLAO / SeqLoRA |
|---|---:|---:|---:|
| Talapas A100/BF16 | 51.0375 ± 1.3629% / -2.2638 ± 1.7875 pp | 43.6469 ± 2.6373% / -13.1557 ± 3.4462 pp | 8,605.84 / 8,253.16 s |
| Kaggle P100/FP16 | 50.3136 ± 0.8334% / -3.6288 ± 1.2187 pp | 45.4982 ± 0.6740% / -11.0725 ± 0.5463 pp | 55,232.89 / 55,329.25 s |

Kaggle SLAO is 0.7240 AA points below Talapas, while Kaggle SeqLoRA is 1.8512
points above it. Both platforms agree on the SLAO-over-SeqLoRA direction and
both place SLAO far above the paper's 37.8% target. The registered numeric
target is therefore not reproduced, although the cross-platform method
ordering is robust under this partial setup. Kaggle is about 6.42×/6.70×
slower for SLAO/SeqLoRA, consistent with the P100-versus-A100 and FP16-versus-
BF16 execution difference rather than a scientific-config change.

The Talapas `slao_merged_b_init` variant remains a separately labeled,
non-faithful comparison to Algorithm 1: it initializes B from the merged state
rather than the previous fine-tuned B factor.

The isolated `repro/kaggle-tpu` branch is an approximate port, not a third
paper-reproduction cell. Its canary
`victorharvey27/slao-tpu-runtime-canary-20260722` verified eight TPU v5 lite
devices plus PyTorch/XLA 2.8.0, then was deleted after download and audit. Its
synthetic one-task PyTorch/XLA LoRA smoke also passed `uv run`, frozen-base,
overfit, accelerator, lifecycle, and sensitive-artifact gates. That synthetic
loss is development evidence and is never substituted for the paper's AA/BWT
metrics; a tiny real Transformers/PEFT compatibility gate is next.
