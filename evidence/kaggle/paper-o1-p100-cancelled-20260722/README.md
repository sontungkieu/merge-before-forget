# Cancelled full P100 attempts

The matched SLAO and SeqLoRA runs were submitted successfully and each
verified one real Tesla P100. Kaggle later returned `CANCEL_ACKNOWLEDGED` for
both after roughly 12.43 hours. Each metrics stream contains nine completed
tasks, followed by a logged start of task 10 and no terminal result or
`run_summary.json`.

The nine-task AA/BWT values in `summary.json` and `partial_metrics.csv` are
diagnostics only. They are neither a 15-task O1 metric nor a paper comparison.
The strict KJO audit correctly fails because cancellation prevented notebook
finalization; a second operational-package audit passes, proving the submit,
status, download, accelerator, archive, and sensitive-artifact evidence is
internally consistent.

The runtime recorded `torch.bfloat16` on a P100. PyTorch 2.6 defines
`torch.cuda.is_bf16_supported(including_emulation=True)` by default, so the old
dtype test accepted an emulated path. The corrected code requires native BF16
support and otherwise selects FP16. Any retry must use a new slug and a commit
containing that correction.

Raw diagnostics remain in the ignored local KJO run directories and on the
private Kaggle notebooks. No checkpoint was downloaded.
