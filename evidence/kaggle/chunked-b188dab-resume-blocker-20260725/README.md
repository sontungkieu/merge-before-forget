# Kaggle chunk checkpoint and resume blocker

The corrected seed-42 task-1--8 runs from source commit
`b188dab065982c80e51af06f43ea7a31018f073e` both finished `COMPLETE`.
Their training summaries are intentionally partial: each reports
`status=checkpointed`, exactly eight completed task rows, one
Tesla P100-PCIE-16GB, and resolved dtype `torch.float16`.

| Method | Partial AA | Partial BWT | Training time | Checkpoint bytes |
|---|---:|---:|---:|---:|
| SLAO | 53.6265% | -2.3635 pp | 31,813.13 s | 22,252,046 |
| SeqLoRA | 47.6026% | -9.1033 pp | 31,889.69 s | 7,419,630 |

Both adapter checkpoints and the structured resume inputs passed the chunk
gate. They were packaged without credentials in the private Kaggle dataset
`anhhaphan/slao-s42-chunk1-b188dab-checkpoints`.

The matched task-9--12 resume notebooks were then submitted, but both failed in
their first `copy_repo` cell before accelerator setup, model loading, or
training. Kaggle exposed only the top-level input entry `datasets`, while the
generated copy cell expected the checkpoint dataset directly at
`/kaggle/input/slao-s42-chunk1-b188dab-checkpoints`. The exact exception was:

```text
FileNotFoundError: Kaggle dataset is not mounted. expected=/kaggle/input/slao-s42-chunk1-b188dab-checkpoints dataset_source=anhhaphan/slao-s42-chunk1-b188dab-checkpoints available_inputs=['datasets']
```

Diagnostics-only downloads retained the complete one-cell KJO log triplets.
The exact sensitive-artifact audits scanned 21 files per failed resume and
found no matches. Strict run-directory audits remain failed, as intended,
because the runtime summary is not successful and the per-run copy of the
dataset manifest/push evidence is absent.

No task-9 training occurred, no scientific result was produced, and no retry
was submitted. A retry requires an explicit operational change that resolves
the actual Kaggle mount path while preserving the verified checkpoint hashes.

