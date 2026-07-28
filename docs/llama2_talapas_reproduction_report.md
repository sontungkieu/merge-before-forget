# Llama-2-7B-chat SLAO reproduction on Talapas

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-28
- Verification Status: VERIFIED for the complete disclosed 12-cell execution
  matrix; NOT REPRODUCED for the paper's numerical AA table
- Version Label: llama2_talapas_validation_v2

## Executive verdict

The disclosed Llama-2-7B-chat SuperNI matrix is complete: SLAO and SeqLoRA,
orders O1 and O2, and seeds 42/43/44 all reached Slurm `COMPLETED 0:0` with
15/15 contiguous task rows. Every cell passed source, model revision, config,
hardware, BF16, finite-metric, checkpoint-loadability, log, and sensitive-file
gates.

This is a **disclosed execution reproduction pass**. It is not an exact paper
reproduction because the paper omits seed identities and several settings, and
its stated data cardinalities conflict with the pinned SAPT source. It is also
a **numerical non-reproduction** under the registered 5% symmetric relative
tolerance: all four order-method AA means and both two-order averages are
outside tolerance.

| Method/order | AA mean ± sample SD (%) | BWT mean ± sample SD (pp) |
|---|---:|---:|
| SLAO O1 | 42.9300 ± 1.1333 | -8.4912 ± 1.8787 |
| SeqLoRA O1 | 32.6915 ± 2.8567 | -23.9379 ± 3.5630 |
| SLAO O2 | 45.8989 ± 1.1714 | -3.7061 ± 1.2608 |
| SeqLoRA O2 | 36.4608 ± 2.0667 | -19.8194 ± 2.2514 |
| SLAO O1/O2 average | 44.4144 ± 1.0495 | -6.0986 ± 1.4012 |
| SeqLoRA O1/O2 average | 34.5761 ± 2.2143 | -21.8786 ± 2.3506 |

SLAO exceeds matched SeqLoRA by 10.2385 AA points on O1, 9.4381 on O2,
and 9.8383 for the two-order average. Its BWT is respectively 15.4468,
16.1133, and 15.7800 points less negative. These are descriptive matched
differences under this disclosed setup, not a causal or exact-paper claim.

## Source and protocol

- Paper: *Merge before Forget: A Single LoRA Continual Learning via Continual
  Merging*, ICLR 2026, arXiv:2512.23017, OpenReview `i1Rj7yU6eF`.
- Scientific source commit:
  `6548c74f9e472087a45f9058ba4f2c9627b00bde`.
- Model: `meta-llama/Llama-2-7b-chat-hf`.
- Pinned revision:
  `f5db02db724555f92da89c216ac04704f23d4590`.
- Dataset lineage: `circle-hit/SAPT` commit
  `52a52b920324c656bdb6dac08e43dc600ba22f21`.
- O1 config SHA-256:
  `5009710dbcdae8035816c043d8065e04f12683f76e65daf83b2d6b0553e1f254`.
- O2 config SHA-256:
  `62996b460d8dba32ce06b2518ed9bd8d5df858da972279d3ae91888514d2f70e`.
- Hardware: one NVIDIA A100 80 GB PCIe MIG `3g.40gb` slice per cell.
- Runtime stack: PyTorch `2.6.0+cu124`, CUDA 12.4, resolved
  `torch.bfloat16`.
- Fixed protocol: published 15-task O1/O2 orders, rank 8 on `q_proj` and
  `v_proj`, LR `5e-5`, 5 epochs, batch 2, accumulation 4, seeds 42/43/44.

The clean-room implementation follows Algorithm 1: QR-normalized A
initialization, next-task B initialization from the previous fine-tuned B,
and the time-weighted merged-B update. No scientific YAML, task order,
sample limit, optimizer setting, or seed was changed during recovery.

## Terminal matrix

| Order | Method | Seed | Job | AA (%) | BWT (pp) | Runtime (h) |
|---|---|---:|---:|---:|---:|---:|
| O1 | SLAO | 42 | `45648618` | 42.4749 | -9.1575 | 3.8121 |
| O1 | SeqLoRA | 42 | `45648619` | 31.8639 | -24.4460 | 3.9012 |
| O1 | SLAO | 43 | `45769278` | 42.0950 | -9.9458 | 3.8400 |
| O1 | SeqLoRA | 43 | `45769279` | 35.8707 | -20.1481 | 4.5447 |
| O1 | SLAO | 44 | `45769280` | 44.2201 | -6.3701 | 4.5834 |
| O1 | SeqLoRA | 44 | `45769281` | 30.3399 | -27.2196 | 4.0684 |
| O2 | SLAO | 42 | `45769282` | 44.6349 | -5.0955 | 3.9529 |
| O2 | SeqLoRA | 42 | `45769283` | 34.2627 | -22.3899 | 3.8749 |
| O2 | SLAO | 43 | `45769284` | 46.1139 | -3.3878 | 3.8689 |
| O2 | SeqLoRA | 43 | `45769285` | 38.3646 | -18.1978 | 4.3106 |
| O2 | SLAO | 44 | `45769286` | 46.9478 | -2.6349 | 4.3283 |
| O2 | SeqLoRA | 44 | `45769287` | 36.7549 | -18.8705 | 3.6636 |

## Artifact gate

For every cell, the audit verified:

- Slurm `COMPLETED`, exit `0:0`;
- summary `status=completed`, exactly 15 ordered metric rows, and finite
  losses, AA, BWT, final score vector, and runtime;
- exact scientific commit, model ID/revision, method, seed, task order, and
  config SHA-256;
- clean submitted Git status;
- one A100 40 GB MIG device and BF16 configuration;
- nonempty adapter checkpoint loadable on CPU with format
  `slao-repro-adapter-v2`, correct method/seed/config/order, and task index 15;
- identity, data-fetch, hardware, Slurm, training, and completion logs;
- no fatal traceback/OOM marker and zero exact credential-pattern findings.

| Job | Checkpoint bytes | Checkpoint SHA-256 |
|---:|---:|---|
| `45648618` | 50,451,486 | `6efbf336a0e8032a49108b9a8a7bea7da9e025b1080e76f926f08f11f36602e2` |
| `45648619` | 16,819,806 | `fe9cedd422eb255488ad938f18769631d03e3a0b038a6e834489e63694fb87c8` |
| `45769278` | 50,451,486 | `a55e597098df007fb573fe01da7f1503d117385979851b30fff9c781a7a28ff4` |
| `45769279` | 16,819,806 | `346af5be15b71d350bfa3f4f2b10b407089e64f02e65984ee24080cde9200911` |
| `45769280` | 50,451,486 | `8fe72edaa8b0ae40f3b4ef45c65472f909b75aa8ab3511713da55bcb36db0341` |
| `45769281` | 16,819,806 | `6e9daa5a449e976b85748e66a3cd22e0e2d899eefdbd0abb6ba758aac213e4ae` |
| `45769282` | 50,451,486 | `76fa66ca46416ab5fe16f01dfbfccd349c5cf2fb3ffb9e76fab38c6684f3ee9d` |
| `45769283` | 16,819,806 | `6d692be3172e9ecb4adc5f9728dc848a3069ea9ceae2e6580091634cff3c9971` |
| `45769284` | 50,451,486 | `c69d6c97020e4cc264b9c629246406c26012ced83a7c81edfcd4d7510120662f` |
| `45769285` | 16,819,806 | `b3a154ff46416ace8846618692be561cdde736b7fdcd50213113c4de4eb6ffbe` |
| `45769286` | 50,451,486 | `9598a23e90ef6acb4e443e7ce374e94f7ec691c62ca1ddc6f9da72b47ffe0fff` |
| `45769287` | 16,819,806 | `d6d9e2ae0c1a871f81514ebdc6f3e13c5e6900de376530485a10e2419a0eb150` |

Checkpoints and prediction dumps remain on GPFS; the repository contains only
compact metadata and hashes.

## Paper-target comparison

The paper values appear to be three-seed aggregates. The registered gate uses
`abs(observed-target) / max(abs(observed), abs(target)) < 0.05`.

| Method/order | Paper AA (%) | Observed mean (%) | Delta (pp) | Symmetric difference | Gate |
|---|---:|---:|---:|---:|---|
| SLAO O1 | 38.7 | 42.9300 | +4.2300 | 9.85% | outside |
| SeqLoRA O1 | 18.4 | 32.6915 | +14.2915 | 43.72% | outside |
| SLAO O2 | 35.7 | 45.8989 | +10.1989 | 22.22% | outside |
| SeqLoRA O2 | 26.8 | 36.4608 | +9.6608 | 26.50% | outside |
| SLAO O1/O2 average | 37.2 | 44.4144 | +7.2144 | 16.24% | outside |
| SeqLoRA O1/O2 average | 22.6 | 34.5761 | +11.9761 | 34.64% | outside |

The matched SLAO-over-SeqLoRA direction agrees with the paper, but no reported
AA target is numerically reproduced under the registered tolerance.

## Order disparity

Final O1/O2 scores were aligned by task identity before computing
`OPD_t`, `MOPD=max_t OPD_t`, and `AOPD=mean_t OPD_t`.

| Method | Seed | MOPD (pp) | AOPD (pp) |
|---|---:|---:|---:|
| SLAO | 42 | 20.0002 | 8.6934 |
| SLAO | 43 | 23.3210 | 9.0955 |
| SLAO | 44 | 25.0221 | 7.4831 |
| SeqLoRA | 42 | 19.3333 | 9.5151 |
| SeqLoRA | 43 | 36.5728 | 12.1353 |
| SeqLoRA | 44 | 25.3562 | 9.4868 |

SLAO MOPD is 22.7811 ± 2.5541 pp and AOPD is
8.4240 ± 0.8393 pp. SeqLoRA MOPD is 27.0875 ± 8.7492 pp and AOPD is
10.3791 ± 1.5210 pp. These are descriptive three-seed summaries; no paper
MOPD/AOPD target is asserted here.

## Failure and recovery history

The first ten expansion jobs `45653523`--`45653532` were preempted after
2--12 tasks. Their logs and task-boundary checkpoints remain preserved, and
their partial metrics are not counted as results.

Jobs `45769278`--`45769287` were fresh task-1 restarts on non-preemptible
`gpu/normal`, each with a unique run label and output directory. They were not
checkpoint resumes, so parent-checkpoint continuity is not applicable. This
recovery choice is disclosed rather than presented as a continuation of the
preempted trajectories.

## Exactness limits

The paper omits exact seed identities, model revision, LoRA alpha/dropout,
sequence lengths, and complete decoding settings. In addition, its stated
1,000 train / 100 validation / 100 test rows per task cannot be obtained from
the cited pinned SAPT source for five selected tasks:

| Task | Available train/validation/test |
|---|---:|
| `task1572_samsum_summary` | 160 / 20 / 20 |
| `task181_outcome_extraction` | 338 / 43 / 43 |
| `task639_multi_woz_user_utterance_generation` | 142 / 18 / 18 |
| `task1590_diplomacy_text_generation` | 126 / 16 / 16 |
| `task073_commonsenseqa_answer_generation` | 975 / 100 / 100 |

The run uses every available pinned row without duplication. Therefore:

- `execution_reproduction = PASS`;
- `numerical_reproduction = NOT_REPRODUCED`;
- evidence class remains `partial`, not `exact`;
- no tuning-to-target or undisclosed resampling was performed.

## Statistical and methodological validation

Environment classification: stochastic, environment-sensitive training.
Mean and sample SD are reported over the pre-registered three seeds. The 5%
criterion is a numerical agreement gate, not a significance or equivalence
test.

### Eleven-type fallacy scan

Coverage: **11/11 checked**.

| Fallacy | Assessment |
|---|---|
| Simpson's paradox | Order/method aggregates are accompanied by seed-level cells and task-level matrices; no hidden subgroup reversal is claimed. |
| Ecological fallacy | No individual-example inference is drawn from task or seed aggregates. |
| Berkson's paradox | Not applicable to the fixed benchmark stream. |
| Collider bias | No covariate-adjusted causal model is used. |
| Base-rate neglect | No sensitivity/specificity or prevalence claim is made. |
| Regression to the mean | Seeds were pre-registered and none was selected by outcome. |
| Survivorship bias | Preempted attempts are retained and disclosed; only complete replacement cells enter aggregates. |
| Look-elsewhere effect | AA, BWT, MOPD/AOPD, seeds, and the 5% gate were fixed before completion. |
| Garden of forking paths | `CAUTION`: paper-omitted settings required disclosed assumptions; no post-hoc tuning was used. |
| Correlation implies causation | Matched differences are descriptive and are not generalized causally. |
| Reverse causality | Not applicable to the controlled training comparison. |

## Supported conclusion

The evidence supports that the clean-room implementations execute the full
disclosed Llama matrix on Talapas, that SLAO is descriptively stronger than
matched SeqLoRA for both orders, and that the resulting values do not reproduce
the paper's numerical AA table. It does not support an exact reproduction,
causal attribution of the gap, or claims about undisclosed paper conditions.
