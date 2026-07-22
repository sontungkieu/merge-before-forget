# Active TPU LoRA smoke

This is a development-only, one-task synthetic LoRA overfit gate. It is not a
paper result. `handoff.json` records the actual Kaggle-canonicalized kernel id,
the pinned source commit, status method, and ignored local KJO run directory so
the job can be resumed safely after a workstation shutdown.

A queued/running state is not accepted as evidence. The gate requires a
downloaded `tpu_lora_smoke.json`, a matching `TpuV5E8` accelerator summary,
complete KJO logs, sensitive-artifact audit, and strict run-directory audit.
