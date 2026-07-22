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

## Hardware/runtime evidence gate

The private, log-only canary
`victorharvey27/slao-tpu-runtime-canary-20260722` was submitted from commit
`9dd2142` with exact shape `TpuV5E8`; pre-submit KJO logging and accelerator
contracts passed. It completed after 2,287 seconds in queue and 161 seconds of
scheduler run time. The instrumented cells took 37.812551 seconds.

Downloaded evidence independently reports eight TPU v5 lite devices from JAX
and eight supported `xla` devices from PyTorch/XLA 2.8.0. The KJO accelerator
summary classifies the evidence as `multi_device_backend`, matches the requested
shape, and reports no accelerator warnings. The strict run-directory audit
passed; the sensitive-artifact audit scanned 188 files with zero findings.
Hashes and evidence pointers are committed under
`evidence/kaggle/tpu-active-canary-20260722/`.

The canary used the declared `logs-only/delete-after-download` retention mode.
After verified evidence was pushed, the remote kernel was deleted at
2026-07-22T10:33:29Z; the post-delete KJO audit also passed.

This passes the hardware/runtime gate only. No model was trained and no
scientific metric is claimed. The next gate is a tiny PyTorch/XLA LoRA smoke
invoked through `uv run`; a 15-task O1 attempt remains prohibited until smoke
timing projects below Kaggle's nine-hour TPU session limit.

The planned smoke is intentionally synthetic and single-device. It verifies a
frozen base weight, nonzero-A/zero-B LoRA initialization, trainable low-rank
factors, one-task overfit, actual XLA execution, and `uv run` invocation. It
does not establish PEFT/Transformers compatibility or justify a paper-table
claim.
