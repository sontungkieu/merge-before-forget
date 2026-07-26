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

This passes the hardware/runtime gate only. No scientific metric is claimed.

The follow-up private smoke
`victorharvey27/slao-tpu-pytorch-xla-lora-smoke-20260722` is intentionally
synthetic and single-device. It verifies a
frozen base weight, nonzero-A/zero-B LoRA initialization, trainable low-rank
factors, one-task overfit, actual XLA execution, and `uv run` invocation. It
completed with both KJO cells passing. Downloaded metrics show loss decreasing
from 8.3334245682 to 0.0262091141 in 80 steps, a 99.6855% reduction, while the
base-weight SHA-256 stayed unchanged. Training took 0.6585 seconds after setup;
the KJO-instrumented notebook section took 122.5640 seconds. Kaggle status
polling missed the running transition, so only the 5,498-second
submit-to-terminal wall time is reported, not an inferred scheduler run time.

The strict run-directory audit passed. Under the declared logs-only
`delete-after-download` policy, the remote smoke kernel was deleted at
2026-07-22T12:17:37Z; the post-delete audit passed and the final
sensitive-artifact scan covered 333 files with zero findings. This is still a
**development smoke**, not a scientific result: it does not establish
PEFT/Transformers compatibility or justify a paper-table claim. The next
evidence gate is a tiny real
Transformers/PEFT checkpoint on PyTorch/XLA before any sequential or 15-task
TPU attempt.

## Transformers/PEFT compatibility gate

The gate is implemented under
`kaggle/tpu/sources/30_transformers_peft_xla_compat.py` with a guarded `.cell`
template. It constructs a tiny random Llama model through Transformers, applies
real PEFT LoRA to `q_proj` and `v_proj`, runs BF16 optimization through the XLA
optimizer step, checks frozen-base and LoRA-only invariants, and performs a
Transformers-base plus PEFT-adapter safetensors save/load round-trip. The gate
also records compile/post-compile timing, XLA memory information when
available, exact package versions, the pinned source commit, checkpoint hashes,
and the maximum logit difference after reload.

The checkpoint is temporary and deleted after its hashes and round-trip result
are recorded, so a future passing run may remain a
`logs-only/delete-after-download` development probe.

The first private attempt,
`victorharvey27/slao-tpu-peft-xla-compat-20260726`, ended in `ERROR`.
Downloaded KJO evidence still verified the requested `TpuV5E8`, eight TPU v5
lite devices, and `runtime_matches_requested=true`. The compatibility cell then
failed in 8.711521 seconds at `import peft` with
`ModuleNotFoundError: No module named 'peft'`. The strict run audit is false
because the terminal run summary correctly records the failed cell; the
sensitive-artifact audit found zero sensitive findings. This is a retained
dependency-provisioning failure, not evidence about PEFT/XLA compatibility or
SLAO performance.

The corrective wrapper provisions exact `transformers 4.51.3`, `peft 0.15.2`,
`accelerate 1.6.0`, and `safetensors 0.5.3` packages into an isolated target
directory with `pip --no-deps`. It verifies `torch/torch_xla 2.8.0` before and
after provisioning, checks the isolated imports and versions, and then runs the
same scientific gate through `uv run --no-project`. The wrapper does not
install, upgrade, or shadow `torch` or `torch_xla`. Until a new terminal run
passes all runtime, invariant, round-trip, timing, memory, KJO, and artifact
audits, the compatibility result remains unresolved.
