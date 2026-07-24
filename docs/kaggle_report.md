# Kaggle evidence report

## Execution contract

- Owners: the first matched pair used `codemaivanngu`; the corrected GPU-only
  retry used `kieutung` after a separate capacity check and reservation. The
  TPU-specific exclusion on `kieutung` was not relaxed.
- Status method: Kaggle Job Ops `check-kernel-status`, backed by
  `kaggle kernels status`, with append-only status history.
- Requested shape: one `NvidiaTeslaP100`; the notebook must independently
  observe one CUDA device and record its model and memory.
- Source: private staged, human-readable notebooks with no embedded or
  Kaggle-secret credentials. Every repository Python entrypoint is called as
  `uv run --no-sync python ...` from a frozen `uv` environment.
- Results are accepted only after terminal status, diagnostics download,
  non-empty structured result validation, lifecycle audit, and sensitive-file
  audit. Checkpoints are not downloaded when metrics and diagnostics suffice.

The reservation-time registry estimate reported 30 GPU-hours remaining, but
Kaggle does not expose authoritative weekly usage here; untracked use was
therefore recorded as possible rather than presenting the estimate as quota.

## Gate history

### Failed infrastructure smoke

`codemaivanngu/slao-repro-smoke-gpu-20260722` (commit `23b1770`) ended in
`ERROR`. Kaggle assigned one P100 to a notebook submitted with the two-T4
shape, and the Hugging Face snapshot resolution omitted both model shards.
The failure happened before model load or scientific training. It was
diagnosed and retained at
`evidence/kaggle/failed-smoke-20260722/summary.json`; no metric was claimed.

### Verified P100 retry

`codemaivanngu/slao-repro-smoke-gpu-p100-r2-20260722` ran commit `4f566c0`
and finished `COMPLETE`. The accelerator probe observed exactly one
Tesla P100-PCIE-16GB, matching the declared shape. The model canary downloaded
and verified both pinned Qwen2.5-3B safetensors shards (6,171,926,992 bytes)
before training. All seven logged notebook cells passed, as did overfit,
three-task synthetic sequence, and checkpoint round-trip gates.

The tiny 8-train/4-test, one-step Samsum smoke scored 25.9655 Rouge-L and took
18.98 seconds. This is development evidence, not a paper cell. KJO measured
13 seconds queued and 261 seconds from first running status to terminal status.
All 35 compact outputs were downloaded, the 45-file sensitive-artifact audit
found zero matches, and the remote kernel was then deleted. Compact evidence
is in `evidence/kaggle/smoke-p100-r2-20260722/`.

## Full matched runs

Both runs use commit `4f566c02b849564ea3b60d01384ac5909395a6a1`, the
pinned Qwen2.5-3B revision, SuperNI O1, seed 42, and identical data, optimizer,
LoRA, and evaluation settings. They differ only in the continual method.

| Method | Kernel ID | Submitted UTC | Current result state |
|---|---|---:|---|
| SLAO | `codemaivanngu/slao-paper-o1-s42-p100-20260722` | 2026-07-21 21:01:26 | `CANCEL_ACKNOWLEDGED`; 9/15 tasks only |
| SeqLoRA | `codemaivanngu/seqlora-paper-o1-s42-p100-20260722` | 2026-07-21 21:01:40 | `CANCEL_ACKNOWLEDGED`; 9/15 tasks only |

Both kernels verified one P100, then ran for 44,743/44,744 seconds and were
cancelled while task 10 was in progress. Their nine completed-task diagnostic
values were SLAO AA 55.9939/BWT -2.2684 and SeqLoRA AA 54.2312/BWT -3.8885.
They are not comparable to the paper's 15-task 37.8 AA target and are not
reported as reproduced results. Compact evidence and hashes are in
`evidence/kaggle/paper-o1-p100-cancelled-20260722/`.

The failure exposed a runtime bug: PyTorch 2.6's BF16 capability query includes
emulation by default, so P100 used `torch.bfloat16` instead of the intended FP16
fallback. The corrected runner requests native BF16 support explicitly. A
corrected retry must use a distinct slug and source commit. The strict KJO
audits remain failed because the cancelled notebooks never wrote
`run_summary.json`; operational evidence audits pass and both sensitive scans
found zero matches. No checkpoint was downloaded.

## Explicit native-FP16 retry

After the failure evidence and dtype regression tests were committed, matched
SLAO and SeqLoRA retries were submitted from commit `3848d02` to
`kieutung/slao-o1-s42-p100-native-fp16-retry` and
`kieutung/seqlora-o1-s42-p100-native-fp16-retry`. Both ended
`CANCEL_ACKNOWLEDGED` at 2026-07-22T22:08:28Z after about 12 hours 13 minutes
running. Each manifest verifies commit `3848d02`, exactly one
Tesla P100-PCIE-16GB, no native BF16, and resolved dtype `torch.float16`.

Each run completed only 11/15 task rows and was training task 12 when Kaggle
cancelled it. The truncated diagnostics are SLAO AA/BWT
51.9522%/-2.1658 pp and SeqLoRA 49.3542%/-5.8862 pp. These values are neither
paper results nor valid comparisons with the paper's 37.8% target or Talapas's
completed 15-task cells. Diagnostics-only download retained 23 files per run,
no checkpoint, and exact credential audits found zero matches across 38 files
per run. Operational-package audits passed; strict completion audits failed on
the absent `run_summary.json` and non-final scientific summary. No scientific
retry was silently submitted.

One intervening local probe omitted the explicit Kaggle CLI path and recorded
`LIST_FAILED` at 12:14:52Z. Later explicit-CLI polls and the registry prove this
was an observer error. KJO's derived status-summary duration therefore ends
too early; the reported 43,999/43,997-second running times are recomputed from
registry timestamps. Compact evidence and hashes are in
`evidence/kaggle/active-fp16-retry-20260722/`. Both artifact-producing remote
kernels remain retained for review.

## Task-boundary chunk attempt

Matched seed-42 SLAO and SeqLoRA task-1--8 chunks ran commit `2b42725` on one
P100 in native FP16. Each training summary reached `status=checkpointed` with
exactly eight metric rows. The partial diagnostics were SLAO AA/BWT
53.6265%/-2.3635 pp and SeqLoRA 47.6026%/-9.1033 pp.

Both notebooks then ended `ERROR` in the post-processing CSV writer. It paired
the 15-task order with the eight-row partial score matrix under strict zip
semantics and raised `ValueError: zip() argument 2 is shorter than argument 1`.
This occurred after the checkpoint-save and JSON-summary code paths, but the
terminal `ERROR` still fails the operational gate. No checkpoint was
downloaded, no resume dataset was created, and no retry was submitted.

Diagnostics-only downloads verified the requested P100, resolved dtype
`torch.float16`, source commit, and eight rows. Both sensitive-artifact audits
had zero findings; both strict KJO audits correctly remain failed. The local
CSV fix is covered by a partial-matrix regression test. Compact evidence is in
`evidence/kaggle/chunk1-postprocess-error-20260723/`.

## Corrected chunk and blocked resume

The CSV fix was rerun from commit `b188dab` on matched seed-42 P100/FP16
kernels. Both task-1--8 notebooks ended `COMPLETE`; each summary reports
`status=checkpointed` and exactly eight rows. SLAO recorded partial AA/BWT
53.6265%/-2.3635 pp in 31,813 seconds, while SeqLoRA recorded
47.6026%/-9.1033 pp in 31,890 seconds. The downloaded adapter checkpoints were
non-empty, hash-verified, and packaged in the private dataset
`anhhaphan/slao-s42-chunk1-b188dab-checkpoints`.

The matched task-9--12 resumes did not reach model code. Both failed in their
first dataset-copy cell because the generated path expected
`/kaggle/input/slao-s42-chunk1-b188dab-checkpoints`, but the runtime exposed
only a top-level `datasets` entry. This is an operational mount-path mismatch,
not a continual-learning outcome. Diagnostics-only downloads retained the
complete failed-cell logs; exact sensitive scans checked 21 files per run with
zero findings. No retry was submitted. Compact hashes and failure evidence are
in `evidence/kaggle/chunked-b188dab-resume-blocker-20260725/`.

## Isolated TPU feasibility track

The additional branch `repro/kaggle-tpu` is classified as an approximate port.
Its downloaded hardware canary verified the requested `TpuV5E8`: JAX observed
eight TPU v5 lite devices and PyTorch/XLA 2.8.0 exposed `xla:0` through
`xla:7`. The KJO strict and sensitive-artifact audits passed, after which the
log-only canary was deleted according to its retention contract.

The follow-up private kernel
`victorharvey27/slao-tpu-pytorch-xla-lora-smoke-20260722` pinned source commit
`04196cc`, invoked the synthetic LoRA entrypoint through `uv run`, and passed
one-task overfit on real XLA while preserving the frozen base. Loss fell from
8.3334245682 to 0.0262091141 in 80 steps. Both KJO cells, the strict lifecycle
audit, the exact accelerator contract, and the sensitive-artifact audit
passed. This remains development evidence rather than a Transformers/PEFT
paper cell; the logs-only kernel was deleted after evidence download under its
declared retention policy. The next TPU gate is a tiny real Transformers/PEFT
checkpoint on PyTorch/XLA.
