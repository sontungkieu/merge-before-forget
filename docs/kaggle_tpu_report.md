# Kaggle TPU report

## Classification

This is an **approximate TPU port** track. The SLAO paper and this repository's
reference runner use PyTorch/PEFT; the paper does not report TPU results or a
TPU implementation. Hardware canaries, tiny development runs, and any future
Tunix/JAX work therefore cannot be relabeled as an exact or partial paper
cell.

## Capacity gate

The live Kaggle Job Ops scan on 2026-07-22 found one locally available TPU
owner, `victorharvey27`, with no live active kernel and a local-registry
estimate of 20 hours remaining. The estimate has unknown accounting confidence
and explicitly warns that untracked usage is possible. `kieutung` remains
blocked because its prior `TpuV5E8` run exposed only CPU.

## Current evidence gate

The private, log-only canary
`victorharvey27/slao-tpu-runtime-canary-20260722` was submitted from commit
`9dd2142` with exact shape `TpuV5E8`; pre-submit KJO logging and accelerator
contracts passed. Its latest recorded state is `QUEUED`, so hardware remains
unverified. Durable resume metadata is in
`evidence/kaggle/tpu-active-canary-20260722/`. No model or scientific metric is
claimed at this gate.
