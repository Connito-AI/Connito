import json
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError

from connito.shared.chain import (
    CHAIN_COMMIT_MAX_BYTES,
    CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS,
    VALIDATOR_COMMIT_MAX_HF_REPO_ID_CHARS,
    MinerChainCommit,
    ValidatorChainCommit,
    commit_status,
    validate_chain_commit_payload,
)
from connito.shared.checkpoints import ChainCheckpoint, ChainCheckpoints
from connito.shared.config import CheckpointCfg, HfCfg
from connito.shared.hf_distribute import (
    get_hf_upload_readiness,
    resolve_default_checkpoint_repo,
    resolve_hf_repo_ids,
    upload_checkpoint_to_hf,
)
from connito.shared.model import fetch_model_from_chain_validator
from connito.shared.cycle import get_validator_seed_from_commit, hydrate_miner_submissions_from_hf
from connito.validator.run import validate_hf_distribution_config


def _make_config(tmp_path: Path):
    return SimpleNamespace(
        chain=SimpleNamespace(netuid=102, hotkey_ss58="validator-hotkey"),
        ckpt=SimpleNamespace(
            validator_checkpoint_path=tmp_path / "validator_checkpoint",
            checkpoint_topk=2,
            checkpoint_path=tmp_path / "checkpoints",
        ),
        hf=SimpleNamespace(token_env_var="HF_TOKEN"),
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
        "connito.shared.hf_distribute.resolve_default_checkpoint_repo",
        lambda token_env_var, default_repo_name: f"resolved/{default_repo_name}",
    )
    hf_cfg = HfCfg(checkpoint_repo=None, token_env_var="HF_TOKEN", default_repo_name="co")

    upload_repo, chain_repo = resolve_hf_repo_ids(hf_cfg)

    assert upload_repo == "resolved/co"
    assert chain_repo == "resolved/co"


def test_hf_cfg_rejects_invalid_default_repo_name():
    try:
        HfCfg(default_repo_name="owner/co")
    except ValidationError as exc:
        assert "default_repo_name" in str(exc)
    else:
        raise AssertionError("expected invalid default_repo_name validation error")


def test_hf_cfg_returns_advertised_repo_as_upload_repo():
    hf_cfg = HfCfg(checkpoint_repo="owner/repo", default_repo_name="co")

    assert hf_cfg.advertised_repo_id("owner/repo") == "owner/repo"


def test_hf_cfg_explicit_checkpoint_repo_is_used_for_upload_and_chain_repo():
    hf_cfg = HfCfg(checkpoint_repo="present42/cycle", token_env_var="HF_TOKEN", default_repo_name="co")

    upload_repo, chain_repo = resolve_hf_repo_ids(hf_cfg)

    assert upload_repo == "present42/cycle"
    assert chain_repo == "present42/cycle"


def test_checkpoint_cfg_restores_download_concurrency_for_compatibility():
    assert CheckpointCfg().download_concurrency == 4


def test_fetch_model_skips_chain_checkpoint_without_hf_metadata(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    config.ckpt.validator_checkpoint_path.mkdir(parents=True)
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="miner-hotkey"))
    subtensor = SimpleNamespace(block=123, get_subnet_owner_hotkey=lambda netuid: "owner-hotkey")
    chain_checkpoint = _make_chain_checkpoint(hf_repo_id=None, hf_revision=None)

    monkeypatch.setattr(
        "connito.shared.model.build_chain_checkpoints_from_previous_phase",
        lambda **kwargs: ChainCheckpoints(checkpoints=[chain_checkpoint]),
    )
    monkeypatch.setattr(
        "connito.shared.model.download_checkpoint_from_hf",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("HF should not be invoked when coords missing")),
    )
    monkeypatch.setattr("connito.shared.model.delete_old_checkpoints", lambda **kwargs: None)
    monkeypatch.setattr("connito.shared.model.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ChainCheckpoint, "validate", lambda self, expert_group_assignment: True)

    result = fetch_model_from_chain_validator(
        current_model_meta=None,
        config=config,
        subtensor=subtensor,
        wallet=wallet,
        expert_group_ids=[0],
        expert_group_assignment={},
    )

    assert result is None


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


def test_validator_seed_prefers_explicit_miner_seed_when_present():
    config = SimpleNamespace(task=SimpleNamespace(exp=SimpleNamespace(group_id=3)))
    commit_a = ValidatorChainCommit(model_hash="a" * 64, global_ver=101, expert_group=3, miner_seed=11)
    commit_b = ValidatorChainCommit(model_hash="b" * 64, global_ver=101, expert_group=3, miner_seed=29)
    neuron_a = SimpleNamespace(hotkey="validator-a")
    neuron_b = SimpleNamespace(hotkey="validator-b")

    seeds = get_validator_seed_from_commit(config, [(commit_a, neuron_a), (commit_b, neuron_b)])

    assert seeds == {"validator-a": 11, "validator-b": 29}


def test_miner_chain_commit_payload_includes_hf_fields_and_stays_within_budget():
    # block and inner_opt omitted deliberately — they're excluded from the
    # serialized payload when left at their defaults so the HF coords fit
    # within the shared 128-byte chain budget.
    commit = MinerChainCommit(
        expert_group=0,
        model_hash="a" * 64,
        global_ver=1234,
        hf_repo_id="user/co-miner",
        hf_revision="abcdef0",
    )
    data_dict, data = validate_chain_commit_payload(commit)
    assert data_dict["r"] == "user/co-miner"
    assert data_dict["rv"] == "abcdef0"
    assert "b" not in data_dict
    assert "i" not in data_dict
    assert len(data.encode()) <= CHAIN_COMMIT_MAX_BYTES
    # Round-trip through JSON produces equivalent fields.
    reparsed = MinerChainCommit.model_validate(json.loads(data))
    assert reparsed.hf_repo_id == commit.hf_repo_id
    assert reparsed.hf_revision == commit.hf_revision


def test_miner_commit_rejects_repo_id_that_exceeds_budget():
    commit = MinerChainCommit(
        expert_group=0,
        model_hash="a" * 64,
        global_ver=1,
        hf_repo_id="x" * (CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS + 1),
        hf_revision="abcdef0",
    )
    try:
        validate_chain_commit_payload(commit)
    except ValueError as exc:
        assert "too long" in str(exc)
    else:
        raise AssertionError("expected miner HF repo id length failure")


def test_miner_and_validator_share_the_same_payload_budget():
    # Sanity: commit_status routes both schemas through the same validator so
    # the chain can't see divergent caps.
    assert CHAIN_COMMIT_MAX_BYTES == 128
    assert CHAIN_COMMIT_MAX_HF_REPO_ID_CHARS == VALIDATOR_COMMIT_MAX_HF_REPO_ID_CHARS


def test_hydrate_miner_submissions_from_hf_writes_assigned_miners_only(tmp_path, monkeypatch):
    submission_dir = tmp_path / "miner_submission"
    submission_dir.mkdir()

    config = SimpleNamespace(
        chain=SimpleNamespace(netuid=102, hotkey_ss58="validator-hotkey"),
        ckpt=SimpleNamespace(miner_submission_path=submission_dir),
        hf=SimpleNamespace(token_env_var="HF_TOKEN"),
        task=SimpleNamespace(exp=SimpleNamespace(group_id=0)),
    )
    subtensor = SimpleNamespace(block=999)

    assigned = ChainCheckpoint(
        uid=7,
        hotkey="miner-assigned",
        global_ver=10,
        model_hash="abcd",
        signed_model_hash="signed",
        expert_group=0,
        ip="127.0.0.1",
        port=8000,
        hf_repo_id="some-user/co-miner",
        hf_revision="abcdef0",
    )
    unassigned = ChainCheckpoint(
        uid=8,
        hotkey="miner-unassigned",
        global_ver=10,
        model_hash="abce",
        signed_model_hash="signed2",
        expert_group=0,
        ip="127.0.0.2",
        port=8001,
        hf_repo_id="some-user/co-miner2",
        hf_revision="abcdef1",
    )
    without_hf = ChainCheckpoint(
        uid=9,
        hotkey="miner-no-hf",
        global_ver=10,
        model_hash="abcf",
        signed_model_hash="signed3",
        expert_group=0,
        ip="127.0.0.3",
        port=8002,
    )

    monkeypatch.setattr(
        "connito.shared.checkpoints.build_chain_checkpoints_from_previous_phase",
        lambda **kwargs: ChainCheckpoints(checkpoints=[assigned, unassigned, without_hf]),
    )

    seen = []

    def fake_download(**kwargs):
        dest_dir = Path(kwargs["dest_dir"])
        dest_dir.mkdir(parents=True, exist_ok=True)
        for fname in kwargs["filenames"]:
            (dest_dir / fname).write_bytes(b"hf-shard")
            seen.append((kwargs["repo_id"], kwargs["revision"], fname))

    monkeypatch.setattr("connito.shared.cycle.download_checkpoint_from_hf", fake_download)

    hydrated = hydrate_miner_submissions_from_hf(
        config=config,
        subtensor=subtensor,
        validator_miner_assignment={"validator-hotkey": ["miner-assigned", "miner-no-hf"]},
    )

    assert hydrated == 1
    assert seen == [("some-user/co-miner", "abcdef0", "model_expgroup_0.pt")]
    # Assigned HF miner got a submission file with the expected naming convention.
    assert (submission_dir / "hotkey_miner-assigned_block_999.pt").exists()
    # Unassigned miner is skipped even though it has HF coords.
    assert not list(submission_dir.glob("*miner-unassigned*"))
    # Miner without HF coords is missing for this round and gets the zero-score penalty.
    assert not list(submission_dir.glob("*miner-no-hf*"))
    # No leftover tmp dirs from atomic-rename path.
    assert not list(submission_dir.glob(".tmp_*"))


def test_hydrate_miner_submissions_skips_when_local_file_already_present(tmp_path, monkeypatch):
    submission_dir = tmp_path / "miner_submission"
    submission_dir.mkdir()
    # Simulate a submission already landed (e.g. by the background download worker)
    # before the hydrator polls.
    (submission_dir / "hotkey_miner-assigned_block_500.pt").write_bytes(b"prior-upload")

    config = SimpleNamespace(
        chain=SimpleNamespace(netuid=102, hotkey_ss58="validator-hotkey"),
        ckpt=SimpleNamespace(miner_submission_path=submission_dir),
        hf=SimpleNamespace(token_env_var="HF_TOKEN"),
        task=SimpleNamespace(exp=SimpleNamespace(group_id=0)),
    )
    subtensor = SimpleNamespace(block=999)

    ckpt = ChainCheckpoint(
        uid=7,
        hotkey="miner-assigned",
        global_ver=10,
        model_hash="abcd",
        signed_model_hash="signed",
        expert_group=0,
        ip="127.0.0.1",
        port=8000,
        hf_repo_id="some-user/co-miner",
        hf_revision="abcdef0",
    )
    monkeypatch.setattr(
        "connito.shared.checkpoints.build_chain_checkpoints_from_previous_phase",
        lambda **kwargs: ChainCheckpoints(checkpoints=[ckpt]),
    )

    called = []

    def fake_download(**kwargs):
        called.append(kwargs)

    monkeypatch.setattr("connito.shared.cycle.download_checkpoint_from_hf", fake_download)

    hydrated = hydrate_miner_submissions_from_hf(
        config=config,
        subtensor=subtensor,
        validator_miner_assignment={"validator-hotkey": ["miner-assigned"]},
    )

    assert hydrated == 0
    assert called == []


def test_validator_seed_defaults_to_zero_when_miner_seed_missing():
    config = SimpleNamespace(task=SimpleNamespace(exp=SimpleNamespace(group_id=3)))
    commit_a = ValidatorChainCommit(model_hash="a" * 64, global_ver=101, expert_group=3)
    commit_b = ValidatorChainCommit(model_hash="b" * 64, global_ver=101, expert_group=3)
    neuron_a = SimpleNamespace(hotkey="validator-a")
    neuron_b = SimpleNamespace(hotkey="validator-b")

    seeds = get_validator_seed_from_commit(config, [(commit_a, neuron_a), (commit_b, neuron_b)])

    assert seeds == {"validator-a": 0, "validator-b": 0}