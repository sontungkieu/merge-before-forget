# Merge before Forget reproduction

Independent reproduction of **Merge before Forget: A Single LoRA Continual
Learning via Continual Merging** (SLAO), ICLR 2026, arXiv:2512.23017,
OpenReview `i1Rj7yU6eF`.

No author code was linked from the paper when this reproduction began. The
implementation in this repository is derived from Algorithm 1 and the paper's
appendix. The `mcp-tool-shop-org/backpropagate` project is not author code and
is not used as a source of truth.

## Evidence classes

- **Exact**: paper-specified model, data, task order, hyperparameters, metric,
  and seed protocol are all matched.
- **Partial**: a paper cell is attempted but at least one disclosed paper
  detail is missing or fewer seeds are run.
- **Approximate**: a checkpoint, dataset, or method-defining setting differs.
- **Development**: synthetic/tiny gate only; never compared as a paper result.

The first paper-sanctioned target is Qwen2.5-3B on SuperNI order 1. It is
classified as **partial** because the paper does not disclose Qwen-specific
seeds, checkpoint revision, LoRA alpha/dropout, or all decoding details. The
smaller Llama-3.2-3B table cell is license-gated for the available credential.

The matched Kaggle P100/FP16 pipelines are complete for the pre-registered
seeds 42/43/44. SLAO reaches 50.3136 ± 0.8334% AA versus
45.4982 ± 0.6740% for SeqLoRA. The SLAO mean is outside the registered 5%
relative tolerance around the paper's 37.8% row, so the run supports the
method-over-baseline direction but does not quantitatively reproduce that
paper value. See [the Kaggle report](docs/kaggle_report.md) and
[the compact three-seed evidence](evidence/kaggle/three-seed-b188dab-20260726/README.md).

## Reproducible commands

All Python entrypoints are invoked through `uv run`:

```bash
uv sync --frozen --extra test
uv run pytest -q
uv run python scripts/dev_gates.py --output-dir outputs/dev-gates
uv run python scripts/fetch_benchmark_data.py --benchmark superni
uv run python -m slao_repro.train \
  --config configs/paper/qwen25_3b_superni_o1.yaml \
  --method slao --seed 42 --run-label qwen25-3b-superni-o1-s42
```

See [the reproduction protocol](docs/reproduction_protocol.md) and
[claim ledger](docs/claim_ledger.md) before interpreting any result.

## Infrastructure branches

- `repro/talapas`: Talapas/Slurm scripts, manifests, job evidence, and report.
- `repro/kaggle`: deterministic Kaggle notebook source, registry evidence,
  downloaded diagnostics, and report.
