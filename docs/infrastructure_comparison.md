# Infrastructure comparison

The Talapas and Kaggle branches were created as independent worktrees from the
same base commit, `3e9bed0d1baeac907eaab21a526053400ac2c405`. Their paper-cell
runs use the same clean-room SLAO implementation, Qwen2.5-3B revision, pinned
SAPT data, O1 task order, seed 42, LoRA settings, optimizer, and metrics.

| Dimension | Talapas | Kaggle |
|---|---|---|
| Branch | `repro/talapas` | `repro/kaggle` |
| Full-run source commit | `f137983960487b501ebedc4eff83257e20ad5c5f` | `4f566c02b849564ea3b60d01384ac5909395a6a1` |
| Scheduler/status evidence | Slurm `squeue`, `scontrol`, and `sacct` | KJO status history backed by Kaggle kernels API |
| GPU | A100 80GB PCIe 3g.40gb MIG slice | one Tesla P100-PCIE-16GB |
| Model dtype | BF16 | FP16 fallback because P100 lacks BF16 |
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

The final section will compare verified AA/BWT, runtime, and absolute/relative
delta to the paper's SLAO Qwen2.5-3B SuperNI O1 target of 37.8% after both
infrastructure audits pass.
