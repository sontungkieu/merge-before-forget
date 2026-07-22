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

Pending submission of a private, log-only `TpuV5E8` runtime canary. The canary
must verify real devices independently through KJO and backend probes, then
report whether PyTorch/XLA is usable. No model or scientific metric is claimed
at this gate.
