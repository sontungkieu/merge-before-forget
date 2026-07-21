## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + run + validate
- Origin Date: 2026-07-22
- Verification Status: ANALYZED
- Version Label: slao_protocol_v1

## Reproduction question

Can a clean-room implementation of SLAO Algorithm 1 reproduce a paper table
cell while retaining a single merged LoRA and outperforming matched SeqLoRA on
the same task stream?

## Primary sources

- Paper and appendix: arXiv:2512.23017 v1 and OpenReview `i1Rj7yU6eF`.
- Standard/long benchmark data lineage: official O-LoRA repository,
  `cmnfriend/O-LoRA` commit `07117e1fc4a5f5ad9308a815a42cee8f46502dc8`.
- SuperNI data/task lineage: official SAPT repository, `circle-hit/SAPT` commit
  `52a52b920324c656bdb6dac08e43dc600ba22f21`.

No SLAO author implementation was available. Third-party SLAO code is excluded
from implementation decisions and may only be used for a post-hoc differential
audit.

## Algorithm contract

For task 1, standard LoRA fine-tuning yields `(B_ft,1, A_ft,1)` and initializes
the merged state. For every task `i >= 2` and each adapted layer:

1. Compute reduced QR of `A_ft,i-1^T`: `Q R = qr(A^T)`.
2. Canonicalize QR signs with `Q <- Q diag(sign(diag(R)))`, replacing an exact
   zero sign by `+1`; initialize `A_ft,i <- Q^T`.
3. Initialize `B_ft,i <- B_ft,i-1` (the last fine-tuned B, not merged B).
4. Fine-tune both A and B on task i.
5. Set `A_merge,i <- A_ft,i`.
6. Set `B_merge,i <- B_merge,i-1 + i^(-1/2)(B_ft,i-B_merge,i-1)`.
7. Evaluate with `(B_merge,i, A_merge,i)` and retain the last fine-tuned state
   separately for initialization of the next task.

## Metrics

- `AA = mean_i a[i,T]`, using normalized exact-match accuracy for the three
  SuperNI classification tasks and Rouge-L F1 for other SuperNI tasks.
- `BWT = mean_{i<T}(a[i,T]-a[i,i])`.
- For multiple orders, `OPD_t = max_r P_t^r - min_r P_t^r`,
  `MOPD=max_t OPD_t`, and `AOPD=mean_t OPD_t`; lower is better.
- The first paper target is Qwen2.5-3B, SuperNI O1, SLAO AA 37.8%.
- Stochastic agreement defaults to less than 5% relative difference, but no
  equivalence verdict is issued from one seed.

## Staged gates

1. Static checks: locked dependency graph, Ruff, compilation, and `git diff --check`.
2. Unit tests: QR/sign, A/B initialization, coefficient, sequential merge,
   serialization, AA/BWT, MOPD/AOPD.
3. Tiny synthetic smoke: finite losses and valid state transitions.
4. One-task overfit: a rank-representable synthetic task must reduce loss by at
   least 95%.
5. Short sequential run: three synthetic tasks, structured matrix and summary.
6. Infrastructure canary: exact GPU is recorded and runtime import/model access
   succeeds.
7. Paper cell/order: SLAO and matched SeqLoRA, initially seed 42 only.
8. Wider comparison: remaining seeds, O2, and an additional merging baseline
   only after gate 7 produces verified artifacts.

## Failure policy

Scientific failures are recorded and diagnosed without silent resubmission.
Infrastructure-only retries may preserve the same run identity when code,
data, seed, and configuration hashes are unchanged. A queued or submitted job
is never counted as a result.

