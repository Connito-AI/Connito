import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from pydantic import ValidationError

from connito.shared.chain import (
    VALIDATOR_COMMIT_MAX_HF_REPO_ID_CHARS,
    ValidatorChainCommit,
    commit_status,
)
from connito.shared.checkpoints import ChainCheckpoint, ChainCheckpoints
from connito.shared.config import CheckpointCfg, HfCfg
from connito.shared.hf_distribute import (
    get_hf_upload_readiness,
    resolve_default_checkpoint_repo,
    upload_checkpoint_to_hf,
)
from connito.shared.model import fetch_model_from_chain_validator
from connito.shared.cycle import get_validator_seed_from_commit
from connito.validator.run import resolve_hf_repo_ids, validate_hf_distribution_config
from connito.validator import server


def _make_config(tmp_path: Path):
    return SimpleNamespace(
        chain=SimpleNamespace(netuid=102, hotkey_ss58="validator-hotkey"),
        ckpt=SimpleNamespace(
            validator_checkpoint_path=tmp_path / "validator_checkpoint",
            checkpoint_topk=2,
            checkpoint_path=tmp_path / "checkpoints",
        ),
        hf=SimpleNamespace(token_env_var="HF_TOKEN"),
        cycle=SimpleNamespace(token=""),
        miner=SimpleNamespace(protocol="http"),
    )


def _make_chain_checkpoint(**overrides):
    base = dict(
        uid=7,
        hotkey="validator-hotkey",
        global_ver=42,
        model_hash="abcd",
        signed_model_hash="signed",
        expert_group=0,
        ip="127.0.0.1",
        port=8000,
        hf_repo_id="owner/repo",
        hf_revision="rev-1",
    )
    base.update(overrides)
    return ChainCheckpoint(**base)


def test_hf_upload_readiness_reports_missing_repo_and_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    ready, reason = get_hf_upload_readiness(repo_id=None, token_env_var="HF_TOKEN")
    assert not ready
    assert "repo not configured" in reason

    ready, reason = get_hf_upload_readiness(repo_id="owner/repo", token_env_var="HF_TOKEN")
    assert not ready
    assert "HF token missing" in reason


def test_upload_checkpoint_to_hf_requires_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()

    try:
        upload_checkpoint_to_hf(ckpt_dir=ckpt_dir, repo_id="owner/repo", token_env_var="HF_TOKEN")
    except RuntimeError as exc:
        assert "HF token missing" in str(exc)
    else:
        raise AssertionError("expected missing token RuntimeError")


def test_resolve_default_checkpoint_repo_uses_authenticated_namespace(monkeypatch):
    class DummyApi:
        def __init__(self, token):
            self.token = token

        def whoami(self):
            return {"name": "alice"}

    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setattr("connito.shared.hf_distribute.HfApi", DummyApi)

    repo_id = resolve_default_checkpoint_repo(token_env_var="HF_TOKEN", default_repo_name="co")

    assert repo_id == "alice/co"


def test_hf_cfg_resolves_default_upload_repo_when_config_omits_repo(monkeypatch):
    monkeypatch.setattr(
        "connito.validator.run.resolve_default_checkpoint_repo",
        lambda token_env_var, default_repo_name: f"resolved/{default_repo_name}",
    )
    config = SimpleNamespace(
        hf=HfCfg(checkpoint_repo=None, token_env_var="HF_TOKEN", default_repo_name="co")
    )

    upload_repo, chain_repo = resolve_hf_repo_ids(config)

    assert upload_repo == "resolved/co"
    assert chain_repo == "resolved/cycle"


def test_hf_cfg_rejects_invalid_default_repo_name():
    try:
        HfCfg(default_repo_name="owner/co")
    except ValidationError as exc:
        assert "default_repo_name" in str(exc)
    else:
        raise AssertionError("expected invalid default_repo_name validation error")


def test_hf_cfg_derives_advertised_repo_from_upload_repo():
    hf_cfg = HfCfg(checkpoint_repo="owner/repo", default_repo_name="co")

    assert hf_cfg.advertised_repo_id("owner/repo") == "owner/cycle"


def test_checkpoint_cfg_restores_download_concurrency_for_compatibility():
    assert CheckpointCfg().download_concurrency == 4


def test_fetch_model_falls_back_to_http_when_hf_download_fails(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    config.ckpt.validator_checkpoint_path.mkdir(parents=True)
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="miner-hotkey"))
    subtensor = SimpleNamespace(block=123, get_subnet_owner_hotkey=lambda netuid: "owner-hotkey")
    chain_checkpoint = _make_chain_checkpoint()

    monkeypatch.setattr(
        "connito.shared.model.build_chain_checkpoints_from_previous_phase",
        lambda **kwargs: ChainCheckpoints(checkpoints=[chain_checkpoint]),
    )
    monkeypatch.setattr("connito.shared.model.download_checkpoint_from_hf", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("hf down")))
    monkeypatch.setattr("connito.shared.model.delete_old_checkpoints", lambda **kwargs: None)
    monkeypatch.setattr(ChainCheckpoint, "validate", lambda self, expert_group_assignment: True)

    seen = []

    def fake_download_model(**kwargs):
        out_path = Path(kwargs["out_dir"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"ok")
        seen.append((kwargs["url"], out_path.name, kwargs["expert_group_id"]))

    monkeypatch.setattr("connito.shared.model.download_model", fake_download_model)

    result = fetch_model_from_chain_validator(
        current_model_meta=None,
        config=config,
        subtensor=subtensor,
        wallet=wallet,
        expert_group_ids=[0, "shared"],
        expert_group_assignment={},
    )

    assert result is chain_checkpoint
    assert seen == [
        ("http://127.0.0.1:8000/get-checkpoint", "model_expgroup_0.pt", 0),
        ("http://127.0.0.1:8000/get-checkpoint", "model_shared.pt", "shared"),
    ]


def test_fetch_model_uses_http_when_hf_metadata_missing(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    config.ckpt.validator_checkpoint_path.mkdir(parents=True)
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="miner-hotkey"))
    subtensor = SimpleNamespace(block=123, get_subnet_owner_hotkey=lambda netuid: "owner-hotkey")
    chain_checkpoint = _make_chain_checkpoint(hf_repo_id=None, hf_revision=None)

    monkeypatch.setattr(
        "connito.shared.model.build_chain_checkpoints_from_previous_phase",
        lambda **kwargs: ChainCheckpoints(checkpoints=[chain_checkpoint]),
    )
    monkeypatch.setattr("connito.shared.model.download_checkpoint_from_hf", lambda **kwargs: (_ for _ in ()).throw(AssertionError("HF should not be used")))
    monkeypatch.setattr("connito.shared.model.delete_old_checkpoints", lambda **kwargs: None)
    monkeypatch.setattr(ChainCheckpoint, "validate", lambda self, expert_group_assignment: True)

    seen = []

    def fake_download_model(**kwargs):
        out_path = Path(kwargs["out_dir"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"ok")
        seen.append(out_path.name)

    monkeypatch.setattr("connito.shared.model.download_model", fake_download_model)

    result = fetch_model_from_chain_validator(
        current_model_meta=None,
        config=config,
        subtensor=subtensor,
        wallet=wallet,
        expert_group_ids=[0],
        expert_group_assignment={},
    )

    assert result is chain_checkpoint
    assert seen == ["model_expgroup_0.pt"]


def test_get_checkpoint_endpoint_serves_local_checkpoint(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    shard = checkpoint_dir / "model_expgroup_0.pt"
    shard.write_bytes(b"checkpoint-bytes")

    server.config = SimpleNamespace(
        chain=SimpleNamespace(hotkey_ss58="validator-hotkey"),
        ckpt=SimpleNamespace(checkpoint_path=tmp_path / "checkpoints"),
    )
    server.subtensor = SimpleNamespace(block=999)
    monkeypatch.setattr(server, "verify_message", lambda **kwargs: True)
    monkeypatch.setattr(server, "construct_block_message", lambda **kwargs: b"msg")
    monkeypatch.setattr(server, "select_best_checkpoint", lambda primary_dir: SimpleNamespace(path=checkpoint_dir, global_ver=42))

    client = TestClient(server.app)
    response = client.request(
        "GET",
        "/get-checkpoint",
        data={
            "target_hotkey_ss58": "validator-hotkey",
            "origin_hotkey_ss58": "miner-hotkey",
            "origin_block": "123",
            "signature": "sig",
            "expert_group_id": "0",
        },
    )

    assert response.status_code == 200
    assert response.content == b"checkpoint-bytes"


def test_get_checkpoint_endpoint_rejects_wrong_target(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_expgroup_0.pt").write_bytes(b"checkpoint-bytes")

    server.config = SimpleNamespace(
        chain=SimpleNamespace(hotkey_ss58="validator-hotkey"),
        ckpt=SimpleNamespace(checkpoint_path=tmp_path / "checkpoints"),
    )
    server.subtensor = SimpleNamespace(block=999)
    monkeypatch.setattr(server, "verify_message", lambda **kwargs: True)
    monkeypatch.setattr(server, "construct_block_message", lambda **kwargs: b"msg")
    monkeypatch.setattr(server, "select_best_checkpoint", lambda primary_dir: SimpleNamespace(path=checkpoint_dir, global_ver=42))

    client = TestClient(server.app)
    response = client.request(
        "GET",
        "/get-checkpoint",
        data={
            "target_hotkey_ss58": "different-validator",
            "origin_hotkey_ss58": "miner-hotkey",
            "origin_block": "123",
            "signature": "sig",
            "expert_group_id": "0",
        },
    )

    assert response.status_code == 403


def test_validator_chain_commit_payload_stays_compact_with_hf_fields():
    class DummySubtensor:
        def __init__(self):
            self.block = 9_999_999
            self.calls = []

        def set_commitment(self, wallet, netuid, data, raise_error=False):
            self.calls.append(
                {
                    "wallet": wallet,
                    "netuid": netuid,
                    "data": data,
                    "raise_error": raise_error,
                }
            )
            return True

    config = SimpleNamespace(chain=SimpleNamespace(netuid=102))
    wallet = SimpleNamespace(name="dummy-wallet")
    subtensor = DummySubtensor()

    status_with_hf = ValidatorChainCommit(
        model_hash="a" * 64,
        global_ver=123456,
        expert_group=7,
        hf_repo_id="owner/cycle",
        hf_revision="53ddbcd",
    )
    status_without_hf = ValidatorChainCommit(
        model_hash="a" * 64,
        global_ver=123456,
        expert_group=7,
    )

    commit_status(config=config, wallet=wallet, subtensor=subtensor, status=status_with_hf)

    assert len(subtensor.calls) == 1
    committed = subtensor.calls[0]
    payload = committed["data"]
    payload_dict = json.loads(payload)
    payload_without_hf = json.dumps(status_without_hf.model_dump(by_alias=True, exclude_none=True), separators=(",", ":"))

    assert committed["netuid"] == 102
    assert committed["raise_error"] is False
    assert payload_dict["r"] == "owner/cycle"
    assert payload_dict["rv"] == "53ddbcd"
    assert "m" not in payload_dict
    assert "s" not in payload_dict
    assert "b" not in payload_dict
    assert "hf_repo_id" not in payload
    assert "hf_revision" not in payload

    payload_bytes = len(payload.encode())
    delta_bytes = payload_bytes - len(payload_without_hf.encode())

    assert payload_bytes <= 128
    assert delta_bytes <= 40


def test_validator_commit_rejects_repo_id_that_exceeds_budget():
    owner = "x" * VALIDATOR_COMMIT_MAX_HF_REPO_ID_CHARS
    config = SimpleNamespace(
        hf=HfCfg(checkpoint_repo=f"{owner}/repo", token_env_var="HF_TOKEN", default_repo_name="co"),
        task=SimpleNamespace(exp=SimpleNamespace(group_id=0)),
    )

    try:
        validate_hf_distribution_config(config)
    except ValueError as exc:
        assert "too long" in str(exc)
    else:
        raise AssertionError("expected HF repo id length failure")


def test_validator_seed_derivation_no_longer_depends_on_removed_miner_seed():
    config = SimpleNamespace(task=SimpleNamespace(exp=SimpleNamespace(group_id=3)))
    commit_a = ValidatorChainCommit(model_hash="a" * 64, global_ver=101, expert_group=3)
    commit_b = ValidatorChainCommit(model_hash="b" * 64, global_ver=101, expert_group=3)
    neuron_a = SimpleNamespace(hotkey="validator-a")
    neuron_b = SimpleNamespace(hotkey="validator-b")

    seeds = get_validator_seed_from_commit(config, [(commit_a, neuron_a), (commit_b, neuron_b)])

    assert set(seeds) == {"validator-a", "validator-b"}
    assert all(isinstance(seed, int) for seed in seeds.values())
    assert seeds["validator-a"] != seeds["validator-b"]