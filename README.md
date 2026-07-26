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

The completed first target is Qwen2.5-3B on SuperNI order 1. The expanded
Llama-2-7B-chat track covers both published SuperNI orders and their matched
SeqLoRA baseline. Both tracks remain **partial** because the paper does not
disclose exact seed identities, checkpoint revisions, LoRA alpha/dropout,
sequence lengths, or all decoding details, and its stated SuperNI cardinality
conflicts with the cited SAPT data.

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
uv run python -m slao_repro.train \
  --config configs/paper/llama2_7b_chat_superni_o1.yaml \
  --method slao --seed 42 --run-label llama2-7b-chat-superni-o1-s42
```

See [the reproduction protocol](docs/reproduction_protocol.md) and
[claim ledger](docs/claim_ledger.md) before interpreting any result.

## Infrastructure branches

- `repro/talapas`: Talapas/Slurm scripts, manifests, job evidence, and report.
- `repro/kaggle`: deterministic Kaggle notebook source, registry evidence,
  downloaded diagnostics, and report.
