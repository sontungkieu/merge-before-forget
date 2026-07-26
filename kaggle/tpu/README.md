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

The next corrective wrapper adds the project-lock versions
`tokenizers 0.21.4` and `huggingface-hub 0.36.2` to the isolated
`pip --no-deps --target` directory alongside exact `transformers 4.51.3`,
`peft 0.15.2`, `accelerate 1.6.0`, and `safetensors 0.5.3`. It asserts the
coupled Kaggle `torch/torch_xla 2.8.0` runtime before and after provisioning,
verifies all isolated imports and exact versions, then invokes the pinned gate
through `uv run --no-project`. It never installs or shadows `torch` or
`torch_xla`. A new terminal Kaggle run is still required before claiming that
this corrective compatibility gate passes.
