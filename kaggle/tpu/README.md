# Kaggle TPU feasibility gate

This branch is an isolated TPU port investigation. It inherits the clean-room
SLAO implementation and the documented P100 failure, but TPU results are
classified as **approximate port evidence**, not an exact paper reproduction.
The paper workflow is PyTorch/PEFT and does not specify TPU, PyTorch/XLA, JAX,
or Tunix behavior.

Gate order:

1. Reserve one live `TpuV5E8` slot using Kaggle Job Ops. Owner `kieutung` is
   blocked unless independently revalidated.
2. Render and submit the human-readable runtime canary from
   `sources/10_runtime_canary.py`, with the KJO accelerator probe and logging
   contracts required.
3. Download `tpu_runtime_canary.json` and KJO diagnostics. Accept the hardware
   gate only if real TPU devices are observed. Accept the PyTorch path only if
   `torch_xla` imports and exposes an XLA TPU device. This gate passed on
   2026-07-22 with eight TPU v5 lite devices and PyTorch/XLA 2.8.0.
4. Render `sources/20_torch_xla_lora_smoke.cell` into a human-readable staged
   notebook. It checks out the pinned source commit and invokes
   `20_torch_xla_lora_smoke.py` through `uv run --no-project`, preserving the
   Kaggle image's matched PyTorch/XLA runtime. The first smoke pins source
   commit `04196cc6ddda861be679f90056f8cf8639606397`.
5. Run `sources/30_transformers_peft_xla_compat.py` through the adjacent
   `.cell` template after replacing its source-commit placeholder with the
   exact pushed commit. This gate uses a tiny, randomly initialized Llama
   configuration through Transformers and PEFT, targets the paper runner's
   `q_proj`/`v_proj` modules, trains only LoRA A/B in TPU BF16, saves a
   Transformers base plus PEFT safetensors adapter, reloads both, and requires
   matching parameter hashes and logits. The checkpoint exists only in a
   temporary directory and is removed after the round-trip.
6. Use the runtime abstraction in `src/slao_repro/runtime.py` and the guarded
   `sources/40_superni_one_task.cell` template for the first real SuperNI
   task. The staged copy must replace its source-commit and run-id
   placeholders, execute with `--runtime xla --max-tasks 1`, preserve every
   other scientific hyperparameter, and retain its CPU-portable adapter
   checkpoint for review.
7. Do not attempt a chunked or 15-task O1 cell until the one-task run is
   terminal, its diagnostics and checkpoint are audited, and its measured
   compile/post-compile/runtime/memory evidence supports a projection below
   Kaggle's nine-hour TPU session limit.

The canary executes no repository training entrypoint and downloads no model.
The TPU LoRA smoke uses `uv run --no-project` deliberately: syncing the primary
lock would replace Kaggle's coupled `torch==2.8.0`/`torch_xla==2.8.0` runtime
with the paper runner's locked `torch==2.6.0`. The source commit, uv version,
system runtime versions, device, and output are all recorded. The primary
PyTorch paper runner remains frozen by `pyproject.toml` and `uv.lock`. Tunix/JAX
is considered only if PyTorch/XLA becomes unavailable; it remains a separate
approximate implementation, never a transparent replacement for the paper
runner.

The first Transformers/PEFT attempt,
`victorharvey27/slao-tpu-peft-xla-compat-20260726`, preserved the real
eight-device `TpuV5E8` runtime but failed before the compatibility logic at
`import peft` because Kaggle did not provide that package. The failed run and
its diagnostics remain retained as an operational provisioning failure, not a
PEFT/XLA or scientific result.

The second private attempt,
`victorharvey27/slao-tpu-peft-xla-compat-v2-20260726`, also preserved the real
eight-device `TpuV5E8` runtime. Its isolated package provisioning reached the
Transformers import, which rejected Kaggle's preinstalled
`tokenizers==0.23.0rc0`; Transformers 4.51.3 requires
`tokenizers>=0.21,<0.22`. This remains a dependency-provisioning failure.

The third private attempt,
`victorharvey27/slao-tpu-peft-xla-compat-v3-20260726`, preserved the same real
eight-device TPU runtime and the coupled `torch/torch_xla 2.8.0` versions.
Transformers then rejected Kaggle's preinstalled `huggingface-hub==1.21.0`
because Transformers 4.51.3 requires `huggingface-hub>=0.30.0,<1.0`. This is
another retained dependency-provisioning failure.

The fourth private attempt,
`victorharvey27/slao-tpu-peft-xla-compat-v4-20260726`, used the corrective
wrapper with the project-lock versions
`tokenizers 0.21.4` and `huggingface-hub 0.36.2` to the isolated
`pip --no-deps --target` directory alongside exact `transformers 4.51.3`,
`peft 0.15.2`, `accelerate 1.6.0`, and `safetensors 0.5.3`. It asserts the
coupled Kaggle `torch/torch_xla 2.8.0` runtime before and after provisioning,
verifies all isolated imports and exact versions, then invokes the pinned gate
through `uv run --no-project`. It never installs or shadows `torch` or
`torch_xla`.

That run completed and passed the compatibility gate on the requested
eight-device `TpuV5E8`. The base hash remained unchanged, the adapter hash
changed, all losses were finite and improved from 4.795597 to 4.581011, and
the Transformers-base plus PEFT-adapter safetensors round-trip reproduced
logits exactly within the 1e-3 tolerance. Compile, post-compile, checkpoint,
and XLA memory evidence are present; both KJO cells, the strict run-directory
audit, and the 64-file sensitive-artifact audit passed with zero findings.
Compact evidence is under
`evidence/kaggle/tpu-active-peft-xla-compat-20260726/`.

This closes only the tiny development compatibility gate. It is approximate
portability evidence, not a SuperNI result, sequential SLAO/SeqLoRA result, or
paper-comparable reproduction. The next gate remains one real SuperNI task
with projected runtime checked against Kaggle's nine-hour TPU session limit.

The local implementation for that next gate now provides explicit
`auto/cpu/cuda/xla` runtime selection, TPU BF16, XLA optimizer stepping and
blocking synchronization, accelerator-aware batch movement, compile versus
post-compile optimizer timing, XLA memory reporting, and atomic CPU-portable
checkpoints. The single-process gate deliberately selects one XLA device while
recording all eight visible devices and `uses_all_visible_devices=false`;
multi-device execution is not claimed.

The first private one-task submission,
`victorharvey27/slao-tpu-superni-one-task-v1-20260726`, verified the real
eight-device `TpuV5E8` runtime but failed before provisioning or training
because staging replaced both the `run_id` assignment and the literal
placeholder used by its guard. The failed diagnostics are retained. This is an
operational staging failure, not a SuperNI or SLAO result.

The second private attempt,
`victorharvey27/slao-tpu-superni-one-task-v2-20260726`, again verified the
requested eight-device `TpuV5E8` runtime and passed isolated
Transformers/PEFT provisioning without shadowing `torch` or `torch_xla`. It
then failed before model preparation or training because `rouge-score`
imported its undeclared-at-the-gate transitive dependency `nltk`, which was
absent from the Kaggle image. Diagnostics and the failed strict audit are
retained; the next source revision adds only the lock-resolved
`nltk==3.10.0` pin to the isolated `--no-deps` target.

The third private attempt,
`victorharvey27/slao-tpu-superni-one-task-v3-20260727`, provisioned the exact
dependency set, prepared the pinned Qwen checkpoint and SAPT data, and reached
the first forward pass of `task1572_samsum_summary`. It then failed before the
first optimizer step because Transformers selected
`torch.utils.checkpoint.checkpoint`, whose PyTorch 2.8 device lookup expects a
`torch.xla` module that the coupled Kaggle runtime does not expose. The
diagnostics-only download retained all KJO cell logs; the focused audit scanned
44 files with zero sensitive findings, and the strict audit is false only
because the run failed. The runtime now routes XLA gradient checkpointing
through `torch_xla.utils.checkpoint.checkpoint`, which preserves XLA RNG state
and applies the PyTorch/XLA optimization barrier. CUDA and CPU keep the
Transformers checkpoint path.

The fourth private attempt,
`victorharvey27/slao-tpu-superni-one-task-v4-20260727`, used that native
checkpoint path and reached the first real task, but its first lazy training
graph terminated with exit status 137 after roughly 3.5 hours and before the
first optimizer/loss evidence. There is no explicit allocator or OOM message,
so this remains a SIGKILL-like resource failure rather than a confirmed OOM.
The diagnostics-only and accelerator evidence are retained; the focused
54-file sensitive audit has zero findings. The next revision preserves the
scientific YAML and accumulation factor while inserting an XLA-only boundary
between accumulated microbatches, using fixed-width XLA collation to avoid
shape recompilation, and logging first-microbatch/optimizer boundaries. CPU
and CUDA behavior is unchanged.
