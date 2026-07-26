# Kaggle three-seed completed result

Matched SLAO and SeqLoRA pipelines completed SuperNI O1 for seeds 42, 43, and
44 from scientific source
`b188dab065982c80e51af06f43ea7a31018f073e`. Each pipeline used task-boundary
chunks 1--8, 9--12, and 13--15 with hash-verified state continuity.

| Method | AA mean ± sample SD | BWT mean ± sample SD | Runtime mean |
|---|---:|---:|---:|
| SLAO | 50.3136 ± 0.8334% | -3.6288 ± 1.2187 pp | 55,232.89 s |
| SeqLoRA | 45.4982 ± 0.6740% | -11.0725 ± 0.5463 pp | 55,329.25 s |

SLAO leads SeqLoRA by 4.8154 AA points and 7.4437 less-negative BWT points.
Its mean is +12.5136 points (+33.10% target-relative) above the paper's 37.8%
row and outside the registered 5% relative tolerance. The scientific result is
therefore a partial reproduction of the method ordering, not a quantitative
reproduction of the reported paper value.

Every final run has 15 contiguous metric rows, 8,874 prediction rows, one
Tesla P100-PCIE-16GB, `torch.float16`, a non-empty final checkpoint, and exact
task-12 parent-hash continuity. The four seed-43/44 final KJO audits pass the
current 1--4 second spacing policy; exact credential scans checked 84 files per
run with zero findings. Seed 42 retains its truthful historical 5--12 versus
1--4 second audit exception. Artifact-producing kernels remain retained.

The four first-attempt seed-43/44 task-9--12 resumes remain recorded as
operational failures before training: their injected repo-copy cell assumed an
ordinary checkpoint dataset mount path that Kaggle did not use. Mountfix-r2
removed only that copy cell and preserved source, configuration, seeds, task
boundaries, and parent hashes.

Talapas evidence at commit `654f6fc` reports SLAO 51.0375 ± 1.3629% and
SeqLoRA 43.6469 ± 2.6373% AA on A100/BF16. Both platforms agree on method
ordering and disagreement with the paper target. The Talapas
`slao_merged_b_init` variant remains non-faithful to Algorithm 1.

See `summary.json` for exact metrics, hashes, audit gates, and cross-platform
deltas.
