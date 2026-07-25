# Kaggle seed-42 completed result

The matched SLAO and SeqLoRA O1 pipelines from scientific source
`b188dab065982c80e51af06f43ea7a31018f073e` completed all 15 tasks. Each
pipeline was split only at task boundaries (1--8, 9--12, and 13--15), with
hash-verified adapter, metric, and prediction state passed between chunks.

| Method | AA | BWT | Cumulative runtime | Final checkpoint |
|---|---:|---:|---:|---:|
| SLAO | 49.6031% | -5.0018 pp | 55,627.25 s | 22,252,878 bytes |
| SeqLoRA | 45.9781% | -10.5277 pp | 55,701.79 s | 7,420,462 bytes |

SLAO is 3.6250 percentage points higher in AA and 5.5260 points less negative
in BWT than the matched SeqLoRA baseline for seed 42. SLAO is also 11.8031
points above the paper's reported 37.8 O1 target. These are single-seed
observations, not the paper's three-seed result: seeds 43 and 44 remain
required before deciding whether the cell reproduces.

Both terminal kernels reported `status=completed`, exactly 15 cumulative
metric rows, the same configuration hash, one Tesla P100-PCIE-16GB, and
resolved dtype `torch.float16`. Their KJO cell logs contain zero failed cells.
The final checkpoints, structured outputs, runtime accelerator probes, parent
checkpoint hashes, and exact credential scans were verified. Sensitive
artifact audits scanned 47 files per run and found zero matches.

KJO v0.7.0 marks each retrospective run-directory audit non-passing because
the notebooks were submitted with the earlier 5--12 second inter-submit delay,
whereas the current policy requires 1--4 seconds. The submit succeeded and
this mismatch does not alter the downloaded scientific artifacts, but the
historical audit result is preserved rather than rewritten. All later
submissions use the current 1--4 second policy.

