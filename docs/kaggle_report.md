# Kaggle evidence report

## Execution contract

- Owner: `codemaivanngu` (selected after safe parsing, live capacity scan, and
  reservation; `kieutung` was explicitly excluded).
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
`kieutung/seqlora-o1-s42-p100-native-fp16-retry`. Both reached `RUNNING` via
the status API. This owner is used for GPU only; the TPU-specific block on
`kieutung` remains in force. Durable resume metadata is in
`evidence/kaggle/active-fp16-retry-20260722/`. No result is claimed until
downloaded artifacts prove all 15 tasks and the resolved FP16 dtype.
