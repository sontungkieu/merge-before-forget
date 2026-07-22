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
   `torch_xla` imports and exposes an XLA TPU device.
4. Only then implement and test a tiny one-task LoRA update through `uv run`.
5. Do not attempt the 15-task O1 cell unless measured smoke runtime projects
   below Kaggle's nine-hour TPU session limit.

The canary executes no repository training entrypoint and downloads no model.
Any later repository Python entrypoint must be invoked through `uv run` from
the frozen project. Tunix/JAX is considered only if real TPU hardware is
verified and PyTorch/XLA is unavailable; it remains a separate approximate
implementation, never a transparent replacement for the paper runner.
