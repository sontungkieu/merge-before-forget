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

## Current verified comparison

Talapas has completed its primary pair: SLAO AA/BWT is 50.3226%/-3.3698 pp
versus SeqLoRA 45.8978%/-10.2305 pp. SLAO is +12.5226 AA points above the
paper target and outside tolerance, so the result is partial and numerically
non-reproducing even though the run itself completed correctly. Its matched
FTBA-MB-style control reaches 50.3737%/-3.2420 pp, a statistically
uninterpretable one-seed edge of 0.0511/0.1278 points over SLAO.

The first Kaggle P100 pair was canceled after about 44.7 ks when each run had
completed only 9/15 tasks. Diagnostics identified a runtime-selection defect:
PyTorch reported emulated BF16 support on the P100, so the jobs used
`torch.bfloat16` without native hardware support. Their provisional 9-task
AA/BWT values (SLAO 55.9939%/-2.2684 pp; SeqLoRA 54.2312%/-3.8885 pp) are
incomplete and are not comparable to either the paper or Talapas. The defect is
fixed by requiring native BF16 support and otherwise selecting FP16. Corrected
native-FP16 retries are tracked as
`kieutung/slao-o1-s42-p100-native-fp16-retry` and
`kieutung/seqlora-o1-s42-p100-native-fp16-retry`; their status and artifacts
must pass the KJO evidence gates before any result is reported.

An additional isolated branch, `repro/kaggle-tpu`, contains a TPU runtime
canary under kernel `victorharvey27/slao-tpu-runtime-canary-20260722`. This is
an approximate-port hardware gate, not a paper experiment: it must verify real
TPU devices and a usable framework before any scientific smoke run is
attempted. The TPU branch descends from the diagnosed Kaggle implementation
rather than the original default-branch commit, and therefore is not one of
the two primary same-base infrastructure branches.
