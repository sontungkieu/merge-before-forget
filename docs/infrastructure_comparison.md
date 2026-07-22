# Infrastructure comparison

The Talapas and Kaggle branches were created as independent worktrees from the
same base commit, `3e9bed0d1baeac907eaab21a526053400ac2c405`. Their paper-cell
runs use the same clean-room SLAO implementation, Qwen2.5-3B revision, pinned
SAPT data, O1 task order, seed 42, LoRA settings, optimizer, and metrics.

| Dimension | Talapas | Kaggle |
|---|---|---|
| Branch | `repro/talapas` | `repro/kaggle` |
| Full-run source commit | `f137983960487b501ebedc4eff83257e20ad5c5f` | initial `4f566c02b849564ea3b60d01384ac5909395a6a1`; corrected retry `3848d02970c568c2c65dacf8d6ff9c2e29a6a69b` |
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

All Qwen O1 results remain **partial reproduction** even if execution succeeds:
the paper omits Qwen-specific seeds, repository revision, LoRA alpha/dropout,
and complete decoding details; its claimed SuperNI sample cardinalities also
conflict with the cited pinned SAPT splits. One seed cannot establish the
paper's three-seed mean. Tiny smoke runs remain **development** evidence, and a
terminal run lacking verified structured artifacts is **failed** or
**pending**, never successful.

The first Kaggle full attempts were cancelled after about 12.43 hours with only
9/15 tasks complete. The corrected native-FP16 retries also hit Kaggle's
session boundary, this time after about 12.22 running hours and 11/15 completed
tasks. Their diagnostic AA/BWT values are SLAO 51.9522%/-2.1658 pp and SeqLoRA
49.3542%/-5.8862 pp, but a truncated 11-task score matrix is not comparable to
the paper or Talapas. Talapas therefore remains the only completed 15-task
pair: SLAO 50.3226%/-3.3698 pp and SeqLoRA 45.8978%/-10.2305 pp. The Kaggle
hardware/dtype correction is verified, while the scientific completion gate
remains failed.

The isolated `repro/kaggle-tpu` branch is an approximate port, not a third
paper-reproduction cell. Its canary
`victorharvey27/slao-tpu-runtime-canary-20260722` verified eight TPU v5 lite
devices plus PyTorch/XLA 2.8.0, then was deleted after download and audit. Its
synthetic one-task PyTorch/XLA LoRA smoke also passed `uv run`, frozen-base,
overfit, accelerator, lifecycle, and sensitive-artifact gates. That synthetic
loss is development evidence and is never substituted for the paper's AA/BWT
metrics; a tiny real Transformers/PEFT compatibility gate is next.
