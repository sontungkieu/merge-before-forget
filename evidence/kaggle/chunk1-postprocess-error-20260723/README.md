# Kaggle task-boundary chunk 1 terminal evidence

The matched seed-42 SLAO and SeqLoRA kernels ran source commit
`2b4272544eca3a540adb6aa128c393584d12b8ba` on one P100 with resolved dtype
`torch.float16`. Both training summaries report `status=checkpointed` and
exactly eight completed task rows.

Kaggle nevertheless ended both notebooks in `ERROR`. After the checkpoint and
JSON summary code paths, the CSV writer paired the full 15-task order with the
eight-row partial score matrix using `zip(..., strict=True)`. Both methods
therefore raised:

```text
ValueError: zip() argument 2 is shorter than argument 1
```

This is a deterministic post-processing defect rather than a training or
accelerator failure. It still fails the operational gate: neither kernel
finished `COMPLETE`, and the remote adapter checkpoints were not downloaded or
accepted as resume inputs. No resume job or scientific retry was submitted.

Diagnostics-only downloads retained the metrics, summaries, KJO cell logs, and
accelerator evidence. Sensitive-artifact audits found zero findings for both
runs. The strict run-directory audits remain failed because the KJO summaries
correctly classify the terminal cell failures.

The local fix limits CSV row labels to the completed prefix of the task order
and has a regression test for a partial score matrix.
