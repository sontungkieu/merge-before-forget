# Active native-FP16 P100 retries

The two matched retries use source commit `3848d02970c568c2c65dacf8d6ff9c2e29a6a69b`
and distinct Kaggle slugs. They were submitted only after the cancelled BF16
emulation attempts were downloaded, audited, and documented. The exact live
identifiers and resume paths are in `handoff.json`.

No result is claimed while status is queued or running. After terminal status,
download diagnostics only, require one real P100, confirm
`hardware.resolved_dtype == "torch.float16"`, validate all 15 metric rows and
AA/BWT, run the KJO and sensitive-artifact audits, and retain no large local
checkpoint.
