# Deterministic Kaggle workflow

The committed files under `kaggle/sources/` are readable source cells. Before
rendering, copy them into a private run directory and replace only the literal
placeholders `__EXPECTED_SHA__`, `__RUN_ID__`, `__METHOD__`, and `__SEED__`.
The staged copies and rendered notebook are fingerprinted by Kaggle Job Ops.

Required KJO sequence (version 0.7.0):

1. `quota-report --accelerator gpu --live` and `scan-capacity` using the safe
   account parser; reserve one eligible owner with `reserve-owners`.
2. Render the notebook from setup, smoke/paper, and collect sources.
3. `stage-notebook-package` with a concrete GPU shape, private visibility,
   logging instrumentation, accelerator probe injection, and both contracts
   required.
4. `audit-staged-notebook`, then `submit-kernel` with registry recording,
   `secret-mode none`, `artifact-mode has-artifacts`, and retention
   `delete-after-download`.
5. Poll with `check-kernel-status`, download diagnostics, summarize the run,
   audit the run directory and sensitive artifacts, then delete the private
   kernel only after required metrics are locally verified.

The notebook clones the exact public branch commit. It bootstraps the pinned
standalone `uv` binary without a Python installer, syncs `uv.lock`, and invokes
every repository Python entrypoint as `uv run --no-sync ...`. Runtime, checkout,
environment, and Hugging Face cache paths live under `/kaggle/temp`; only compact
evidence is written under `/kaggle/working`. Before scientific training, the
model preparation canary downloads the public Qwen snapshot with bounded,
visible transport retries and verifies every shard named by the safetensors
index. The checkpoint requires no Kaggle Secret. Large adapter checkpoints are
not needed locally; KJO downloads only diagnostics and structured metrics.
