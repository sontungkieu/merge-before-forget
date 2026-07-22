# TPU runtime canary evidence

The exact private Kaggle slug, source commit, registry, and local run directory
are stored in `handoff.json`. The terminal status alone was not accepted as
hardware evidence. Downloaded `KJO_ACCELERATOR_SUMMARY` and
`tpu_runtime_canary.json` independently verified eight TPU v5 lite devices,
JAX TPU, and PyTorch/XLA TPU. `summary.json` records the compact result and
cryptographic hashes of the ignored raw artifacts and audit reports.

After this evidence was pushed, the log-only remote kernel was deleted through
the KJO lifecycle wrapper at 2026-07-22T10:33:29Z. The post-delete audit passed;
its hashes are included in `summary.json`.

This passes only the hardware/runtime feasibility gate. It contains no model
training and no scientific SLAO metric, and remains approximate-port evidence.
