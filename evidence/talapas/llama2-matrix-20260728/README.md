# Llama-2 Talapas matrix audit

This compact bundle records the terminal 2026-07-28 audit of the
Llama-2-7B-chat SLAO/SeqLoRA SuperNI matrix.

- All 12 disclosed cells (two orders, two methods, seeds 42/43/44) completed
  15/15 tasks and passed provenance, checkpoint-loadability, finite-metric,
  log, Slurm, and sensitive-artifact gates.
- Ten first expansion attempts were preempted; their partial checkpoints stay
  on GPFS as infrastructure evidence and are not interpreted as results.
- Jobs `45769278`--`45769287` are fresh non-preemptible replacements, not
  checkpoint resumes.
- The execution reproduction passes. The numerical paper-table reproduction
  does not: all six order/average AA comparisons are outside the registered 5%
  symmetric relative tolerance.
- Exact run-level values, checkpoint hashes, aggregates, and limitations are
  in `summary.json`.
- No credential values, adapter tensors, or prediction dumps are stored here.

See `docs/llama2_talapas_reproduction_report.md` for interpretation.
