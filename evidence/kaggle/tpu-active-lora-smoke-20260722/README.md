# Completed TPU LoRA smoke

This is a development-only, one-task synthetic LoRA overfit gate. It is not a
paper result. `handoff.json` records the actual Kaggle-canonicalized kernel id,
the pinned source commit, status method, and ignored local KJO run directory so
the job can be resumed safely after a workstation shutdown.

The kernel completed and its diagnostics were downloaded. The verified
`tpu_lora_smoke.json` reports 80 optimization steps on `xla:0`, loss falling
from 8.3334245682 to 0.0262091141 (ratio 0.0031450593), and identical frozen
base-weight hashes before and after training. KJO observed eight TPU v5 lite
devices matching `TpuV5E8`; both instrumented cells passed, the sensitive
artifact audit found zero findings, and the strict run-directory audit passed.
The logs-only kernel was then deleted under its declared
`delete-after-download` policy; the post-delete audit also passed with 333
files scanned and zero sensitive findings.

This passes only the synthetic development gate. It does not establish
Transformers/PEFT compatibility and is not a scientific result or paper-table
reproduction. `summary.json` records the compact evidence and raw artifact
hashes; detailed downloaded diagnostics remain in the ignored local KJO run
directory.
