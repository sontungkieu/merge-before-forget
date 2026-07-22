# Active TPU runtime canary

The exact private Kaggle slug, source commit, registry, and local run directory
are stored in `handoff.json`. Queued/running status is not hardware evidence.
Only downloaded `KJO_ACCELERATOR_SUMMARY` and `tpu_runtime_canary.json` may pass
the gate. After a verified diagnostics download and sensitive-artifact audit,
the log-only remote kernel may be deleted through the KJO lifecycle wrapper.
