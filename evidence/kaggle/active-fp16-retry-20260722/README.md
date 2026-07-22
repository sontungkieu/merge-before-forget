# Terminal native-FP16 P100 retries

The matched retries used source commit
`3848d02970c568c2c65dacf8d6ff9c2e29a6a69b`. Both reached terminal
`CANCEL_ACKNOWLEDGED` at 2026-07-22T22:08:28Z after Kaggle ran them for about
12 hours 13 minutes. They are terminal, incomplete scientific failures rather
than reproduced paper cells.

| Method | Completed rows | Diagnostic AA | Diagnostic BWT | Last metric time |
|---|---:|---:|---:|---:|
| SLAO | 11/15 | 51.9522% | -2.1658 pp | 39,181.58 s |
| SeqLoRA | 11/15 | 49.3542% | -5.8862 pp | 39,194.91 s |

The logs show task 12, `task511_reddit_tifu_long_text_summarization`, starting
but not completing. The 11-row AA/BWT values above are useful only for
diagnosing progress; they cannot be compared with the paper's 15-task SLAO O1
target of 37.8% or with the completed Talapas results.

Both manifests verify the required source commit, one
`Tesla P100-PCIE-16GB`, no native BF16, and resolved dtype `torch.float16`.
Diagnostics-only download retained 23 compact files per run and no checkpoint.
The exact credential-value audits scanned 38 files per run and found zero
matches. The operational-package audits passed. The strict completion audits
failed because cancellation prevented `run_summary.json` and the final
scientific summary from being written.

One earlier local poll recorded `LIST_FAILED` at 12:14:52Z because it omitted
the pinned Kaggle CLI path. Later explicit-CLI polls proved that both kernels
were still running. Consequently, KJO's derived `status_summary.json` freezes
its terminal-duration fields at that observer error. The terminal status and
wall times in `summary.json` are recomputed from the append-only registry's
submission, first-running, and true terminal timestamps; the raw history is
preserved unchanged.

Structured evidence is in `summary.json`, row-level diagnostics in
`partial_metrics.csv`, and lifecycle identifiers in `handoff.json`. The remote
artifact-producing kernels remain retained for review; they were not deleted
or retried.
