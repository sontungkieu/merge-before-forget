# Kaggle paper-run shutdown handoff

The two private Kaggle kernels in `handoff.json` execute remotely and continue
when the workstation or WSL is shut down. This handoff contains no credential
or secret value.

On resumption:

1. Use Kaggle Job Ops 0.7.0 and safely re-materialize the selected owner's
   temporary Kaggle configuration from `/home/tung/all-kaggle.json`; the old
   `/tmp` CLI/config paths are intentionally not assumed to survive shutdown.
2. Run `check-kernel-status` for each exact `kernel_id`, using its
   `.secrets/kaggle-runs/<run_id>` directory and the persistent local registry
   `.secrets/kaggle_notebooks.jsonl`.
3. If terminal, download diagnostics and structured result artifacts while
   excluding adapter checkpoints. Validate accelerator shape, all 15 metric
   rows, final summary, AA/BWT recomputation, source commit, and KJO cell logs.
4. Run `summarize-run-dir`, the sensitive-artifact audit, and
   `audit-run-dir`. Only then update reports or claim a result. Delete a remote
   kernel only after its verified compact evidence is locally durable.

At shutdown, both kernels had 43 successful status observations and last
reported `RUNNING`; no full-run scientific result had yet been downloaded.
