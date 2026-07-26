# Llama-2 canonical-checkpoint blocker

- Model: `meta-llama/Llama-2-7b-chat-hf`
- Revision: `f5db02db724555f92da89c216ac04704f23d4590`
- SuperNI order/seed: O1 / 42
- Hardware: one NVIDIA A100 80GB PCIe MIG 3g.40gb slice
- Job `45648515`: `FAILED`, exit `1:0`, elapsed `00:01:14`, source
  `7ddbdcaa70fa1f9360683efa3d69fc2f98969c7b`
- Job `45648524`: `FAILED`, exit `1:0`, elapsed `00:01:07`, source
  `8471d851cbd4de2c3fb704549003cb6a6d5bdb78`

Both jobs passed Ruff, 20 unit tests, and the synthetic development gates
before the real model-load request returned HTTP 403 `GatedRepoError`. Job
`45648524` included the credential-path integration fix. A separate
login-node `hf_hub_download` of the pinned `config.json`, using the same
private token path, returned the same error. This establishes an account-level
authorization blocker rather than a training or scheduler failure.

The complete stdout, stderr, Slurm records, hardware records, and failed-run
manifests remain under the private Talapas evidence root. No credential value
was copied into this bundle. No training began, and these jobs are not
scientific results.
