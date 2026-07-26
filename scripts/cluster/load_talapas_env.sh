#!/usr/bin/env bash

set -euo pipefail

secrets_file="${TALAPAS_SECRETS_ENV:?TALAPAS_SECRETS_ENV must select a private profile}"
expected_account_override="${SLAO_HF_EXPECTED_ACCOUNT:-}"
if [[ "${secrets_file}" != /* ]]; then
  printf 'TALAPAS_SECRETS_ENV must be an absolute path\n' >&2
  return 64 2>/dev/null || exit 64
fi
if [[ ! -f "${secrets_file}" || ! -r "${secrets_file}" ]]; then
  printf 'selected Talapas secrets profile is not a readable file\n' >&2
  return 2 2>/dev/null || exit 2
fi
if find "${secrets_file}" -maxdepth 0 -perm /077 -print -quit | grep -q .; then
  printf 'selected Talapas secrets profile must not grant group/other permissions\n' >&2
  return 2 2>/dev/null || exit 2
fi

# The selected profile, rather than the caller or a shared service cache, owns
# these variables.
unset HF_TOKEN HF_TOKEN_PATH HF_EXPECTED_ACCOUNT
case "${-}" in
  *a*) had_allexport=1 ;;
  *) had_allexport=0 ;;
esac
set -a
# shellcheck disable=SC1090
source "${secrets_file}"
if [[ "${had_allexport}" == "0" ]]; then
  set +a
fi

: "${HF_TOKEN:?HF_TOKEN is missing from the selected Talapas secrets profile}"
if [[ -n "${expected_account_override}" ]]; then
  export HF_EXPECTED_ACCOUNT="${expected_account_override}"
fi
: "${HF_EXPECTED_ACCOUNT:?HF_EXPECTED_ACCOUNT is missing from the selected profile}"
