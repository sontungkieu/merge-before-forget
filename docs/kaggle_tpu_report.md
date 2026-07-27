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

The second private attempt,
`victorharvey27/slao-tpu-peft-xla-compat-v2-20260726`, ended in `ERROR` after
the isolated package install. The real `TpuV5E8` probe again passed with eight
devices, but Transformers 4.51.3 rejected Kaggle's preinstalled
`tokenizers==0.23.0rc0` because it requires `tokenizers>=0.21,<0.22`.
Diagnostics-only download succeeded, the sensitive-artifact audit covered 59
files with zero findings, and the strict audit failed only because the run
summary correctly records the failed compatibility cell.

The third private attempt,
`victorharvey27/slao-tpu-peft-xla-compat-v3-20260726`, ended in `ERROR` after
the isolated install. Runtime evidence again verified the requested
eight-device `TpuV5E8` and coupled `torch/torch_xla 2.8.0`. Transformers then
rejected Kaggle's preinstalled `huggingface-hub==1.21.0` because its declared
range is `huggingface-hub>=0.30.0,<1.0`. Diagnostics-only download succeeded;
the strict audit correctly remains false for the failed compatibility cell.
The first broad sensitive scan reported two medium findings, both the literal
documentation placeholder `KAGGLE_API_TOKEN=<redacted>` copied into capacity
authentication diagnostics, not credential values. The focused scan of the
staged notebook, submitted archive, status, download, and runtime output
covered 56 files with zero findings.

The fourth private attempt,
`victorharvey27/slao-tpu-peft-xla-compat-v4-20260726`, used the corrective
wrapper with the project-lock versions
`tokenizers 0.21.4` and `huggingface-hub 0.36.2` to the isolated target
alongside exact `transformers 4.51.3`, `peft 0.15.2`, `accelerate 1.6.0`, and
`safetensors 0.5.3`, all installed with `pip --no-deps`. It verifies
`torch/torch_xla 2.8.0` before and after provisioning, checks the isolated
imports and versions, and then runs the same scientific gate through
`uv run --no-project`. The wrapper does not install, upgrade, or shadow
`torch` or `torch_xla`.

The run completed from pinned source commit
`6e9eb0a580f0a6d7979d199175758804c0485714`. KJO verified the requested
`TpuV5E8`, eight TPU v5 lite devices, and
`runtime_matches_requested=true`. Runtime versions were
`torch==2.8.0+cpu`, `torch_xla==2.8.0`, `transformers==4.51.3`,
`peft==0.15.2`, `accelerate==1.6.0`, and `safetensors==0.5.3`. The coupled
PyTorch/XLA versions were unchanged by isolated provisioning.

The tiny BF16 PEFT run passed all model invariants: only the four LoRA A/B
weights for `q_proj` and `v_proj` were trainable, A was initially nonzero, B
was initially zero, the frozen-base hash was unchanged, and the adapter hash
changed. All losses were finite and improved from 4.795597 to 4.581011 over 12
steps. The Transformers-base plus PEFT-adapter safetensors checkpoint
round-trip preserved both parameter hashes and reproduced logits with maximum
absolute difference 0.0 under a 1e-3 tolerance.

The initial-forward compile measurement was 0.240804 seconds, the first
optimizer step 0.345272 seconds, the median post-first optimizer step 0.012512
seconds, checkpoint save 0.016249 seconds, and load plus inference 0.173327
seconds. XLA memory evidence reports 1,943,552 bytes used and a 1,944,064-byte
peak after the gate against a 16,909,336,576-byte limit. Both KJO cells passed
in 156.598949 seconds. Diagnostics-only download, the strict run-directory
audit, and the sensitive-artifact audit all passed; the latter scanned 64
operational files with zero findings.

This passes the development Transformers/PEFT-on-XLA compatibility gate only.
It is **approximate portability evidence**, not a SuperNI metric, sequential
SLAO/SeqLoRA result, or paper-comparable result, and it must not be mixed into
the matched P100/A100 three-seed aggregate. Compact evidence is committed under
`evidence/kaggle/tpu-active-peft-xla-compat-20260726/`. A single real SuperNI
task with compile/runtime/memory measurement remains the next gate before any
chunked or 15-task TPU run is proposed.

## One-task SuperNI runtime port

The local TPU branch now has a minimal runtime abstraction under
`src/slao_repro/runtime.py`. The paper YAML remains unchanged; the runner adds
an explicit `--runtime xla` selection. On XLA it preserves configured BF16,
uses the coupled runtime's optimizer step, blocks on device completion for
timing, moves batches through the selected accelerator, records XLA memory
before and after each task, and writes atomic CPU-portable adapter
checkpoints. CUDA and CPU retain their prior default `--runtime auto`
selection.

TPU output is fail-closed as `evidence_class=approximate_portability`,
`paper_comparable=false`, and `result_is_paper_cell=false`, even if a future
TPU invocation covers the full task order. The current single-process
implementation selects one XLA device and explicitly records that it does not
use all eight visible devices; no multi-device speedup is claimed.

`kaggle/tpu/sources/40_superni_one_task.cell` is the guarded source for the
next remote gate. It provisions exact Transformers/PEFT dependencies into an
isolated target without installing or shadowing `torch` or `torch_xla`,
checks out an exact staged source commit, downloads the pinned Qwen and SAPT
inputs, and runs SLAO seed 42 with `--runtime xla --max-tasks 1`. It does not
override epochs, sample counts, optimizer steps, batch sizes, or other
scientific hyperparameters. Its terminal validator requires the real
eight-device TPU topology, one task-complete row, finite timing and memory
evidence, a CPU-portable checkpoint, and approximate-only result labels.

The first private one-task attempt,
`victorharvey27/slao-tpu-superni-one-task-v1-20260726`, ended in `ERROR`.
Downloaded KJO evidence verified the requested `TpuV5E8`, eight TPU v5 lite
devices, and `runtime_matches_requested=true`, but the gate stopped before
dependency provisioning or training. The staging substitution had replaced
both the `run_id` assignment and the literal placeholder used by its guard, so
the staged notebook incorrectly emitted `run_id must be replaced in the staged
notebook`. Diagnostics-only download succeeded; the focused operational
evidence audit covered 61 files with zero findings. A broader run-directory
scan retained two lexical findings from the redacted Kaggle API-token
assignment shown in CLI help text inside capacity diagnostics, not secret
values. The strict audit correctly remains false because the run cell failed.
The second attempt,
`victorharvey27/slao-tpu-superni-one-task-v2-20260726`, also ended in
`ERROR`. Its downloaded accelerator evidence again passed:
`runtime_matches_requested=true`, eight visible TPU v5 lite devices, and a
matching `TpuV5E8` device-count hint. Isolated dependency provisioning
preserved the coupled `torch==2.8.0` and `torch_xla==2.8.0` runtime, but the
first repository import stopped at `ModuleNotFoundError: No module named
'nltk'` through `rouge-score`; model preparation and training never started.
The diagnostics-only download promoted 10 files (144,298 bytes), the focused
48-file sensitive-artifact audit reported zero findings, and the strict run
audit correctly failed because the run summary was not successful. The
minimal follow-up adds only the lock-resolved `nltk==3.10.0` pin to the
isolated `pip --no-deps --target` provisioner. There is still no real SuperNI
TPU metric or runtime projection.

The third attempt,
`victorharvey27/slao-tpu-superni-one-task-v3-20260727`, also ended in
`ERROR`, but progressed through exact dependency verification, the pinned
Qwen model preparation, SAPT data preparation, model placement, and LoRA
construction. It started `task1572_samsum_summary` with 160 training samples
and failed on the first forward pass before any optimizer step. The exact
traceback is `AttributeError: module 'torch' has no attribute 'xla'` from
PyTorch 2.8 activation checkpointing: Transformers 4.51.3 selected
`torch.utils.checkpoint.checkpoint`, which asks PyTorch for a registered
device module for the XLA tensor type. The requested `TpuV5E8`, eight visible
TPU v5 lite devices, coupled `torch/torch_xla 2.8.0`, and all pinned
Transformers/PEFT dependencies were verified. Diagnostics-only download
promoted 10 files (195,128 bytes); the focused 44-file sensitive-artifact
audit reported zero findings, and the strict run audit correctly failed only
because the run summary is unsuccessful.

PyTorch/XLA 2.8 ships its own `torch_xla.utils.checkpoint.checkpoint`
implementation for this boundary. It preserves XLA RNG state during
recomputation and inserts the XLA optimization barrier before backward. The
minimal runtime follow-up keeps the YAML gradient-checkpointing setting
enabled but replaces only the checkpoint function on XLA; CUDA and CPU retain
the Transformers/PyTorch path. There is still no completed SuperNI metric or
runtime projection.

The fourth attempt,
`victorharvey27/slao-tpu-superni-one-task-v4-20260727`, reached the first
task with the XLA-native checkpoint function but terminated with exit status
137 before the first optimizer step or finite loss was materialized. The KJO
cell ran for 12,633.998270 seconds. No Python traceback, XLA allocator error,
or explicit out-of-memory message preceded the termination, so the retained
classification is a SIGKILL-like resource termination during the first lazy
training graph, not a proven TPU OOM. The requested `TpuV5E8`, all eight TPU
v5 lite devices, exact package versions, and pinned source commit
`0ae2f3fdeb79d88c88e56f9e0498cb693ae2d0e8` were verified.
Diagnostics-only download promoted 10 files (159,968 bytes), the 54-file
sensitive-artifact audit found zero findings, and the strict audit failed only
because the scientific run summary was unsuccessful.

The failed loop had no XLA step boundary between the four accumulated
microbatches, allowing the lazy runtime to trace the whole accumulation window
before its first optimizer barrier. This matches the resource-growth pattern
documented in PyTorch/XLA issue
[`#3593`](https://github.com/pytorch/xla/issues/3593). PyTorch/XLA also
documents that variable input shapes cause recompilation and recommends fixed
padding where possible. The next minimal portability revision therefore keeps
the paper YAML, gradient accumulation, batch sizes, epochs, task order, and
sample counts unchanged while adding an XLA-only blocking boundary after each
non-optimizer microbatch and fixed-width XLA collation. It also emits explicit
first-microbatch and first-optimizer boundaries so a future failure can be
localized without inferring an unlogged allocator cause. CPU and CUDA
collation and synchronization behavior remain unchanged.
