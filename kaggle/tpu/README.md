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
6. Do not attempt a real SuperNI task or the 15-task O1 cell until the
   Transformers/PEFT gate passes and the measured one-task runtime projects
   below Kaggle's nine-hour TPU session limit.

The canary executes no repository training entrypoint and downloads no model.
The TPU LoRA smoke uses `uv run --no-project` deliberately: syncing the primary
lock would replace Kaggle's coupled `torch==2.8.0`/`torch_xla==2.8.0` runtime
with the paper runner's locked `torch==2.6.0`. The source commit, uv version,
system runtime versions, device, and output are all recorded. The primary
PyTorch paper runner remains frozen by `pyproject.toml` and `uv.lock`. Tunix/JAX
is considered only if PyTorch/XLA becomes unavailable; it remains a separate
approximate implementation, never a transparent replacement for the paper
runner.

The Transformers/PEFT gate does not install or replace runtime ML packages. It
fails unless the native environment exposes the exact compatibility tuple
`torch/torch_xla 2.8.0`, `transformers 4.51.3`, `peft 0.15.2`,
`accelerate 1.6.0`, and `safetensors 0.5.3`. A missing or mismatched package is
an operational compatibility failure to diagnose explicitly, not permission to
silently replace the coupled XLA runtime.
