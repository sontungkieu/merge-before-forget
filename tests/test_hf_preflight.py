from pathlib import Path

import pytest
import yaml

from slao_repro import hf_preflight


def _write_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "id": "meta-llama/Llama-2-7b-chat-hf",
                    "revision": "pinned-revision",
                }
            }
        ),
        encoding="utf-8",
    )


def test_verify_hugging_face_access_checks_identity_and_exact_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    observed: dict[str, str] = {}

    monkeypatch.setattr(hf_preflight, "whoami", lambda *, token: {"name": "approved-user"})

    def fake_download(*, repo_id: str, filename: str, revision: str, token: str) -> str:
        observed.update(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=token,
        )
        return "/private/cache/config.json"

    monkeypatch.setattr(hf_preflight, "hf_hub_download", fake_download)

    result = hf_preflight.verify_hugging_face_access(
        config_path=config_path,
        expected_account="approved-user",
        token="private-token",
    )

    assert result == {
        "status": "ok",
        "account": "approved-user",
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "revision": "pinned-revision",
    }
    assert observed == {
        "repo_id": "meta-llama/Llama-2-7b-chat-hf",
        "filename": "config.json",
        "revision": "pinned-revision",
        "token": "private-token",
    }


def test_verify_hugging_face_access_rejects_wrong_identity_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    monkeypatch.setattr(hf_preflight, "whoami", lambda *, token: {"name": "shared-user"})

    def unexpected_download(**kwargs: str) -> str:
        raise AssertionError("download must not run after an identity mismatch")

    monkeypatch.setattr(hf_preflight, "hf_hub_download", unexpected_download)

    with pytest.raises(RuntimeError, match="identity mismatch"):
        hf_preflight.verify_hugging_face_access(
            config_path=config_path,
            expected_account="approved-user",
            token="private-token",
        )
