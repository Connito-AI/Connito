from types import SimpleNamespace

from connito.shared.cycle import hydrate_miner_submissions_from_hf


def _make_config(tmp_path):
    return SimpleNamespace(
        chain=SimpleNamespace(hotkey_ss58="validator-hotkey"),
        ckpt=SimpleNamespace(miner_submission_path=tmp_path),
        task=SimpleNamespace(exp=SimpleNamespace(group_id=0)),
        hf=SimpleNamespace(token_env_var="HF_TOKEN"),
    )


def _make_chain_checkpoint(hotkey: str):
    return SimpleNamespace(
        hotkey=hotkey,
        uid=68,
        hf_repo_id="James265988/co10",
        hf_revision="a6ff8f7",
    )


def test_hydrate_miner_submissions_from_hf_replaces_stale_local_files(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    subtensor = SimpleNamespace(block=8040563)
    stale_file = tmp_path / "hotkey_5DeoK1KCdM3a37ZY57BZMu6Y1psfQBBQ7buuho33czRMDBxT_block_8039115.pt"
    stale_file.write_bytes(b"stale")

    monkeypatch.setattr(
        "connito.shared.checkpoints.build_chain_checkpoints_from_previous_phase",
        lambda **kwargs: SimpleNamespace(checkpoints=[_make_chain_checkpoint("5DeoK1KCdM3a37ZY57BZMu6Y1psfQBBQ7buuho33czRMDBxT")]),
    )

    downloaded = []

    def fake_download_checkpoint_from_hf(**kwargs):
        downloaded.append(kwargs)
        (kwargs["dest_dir"] / "model_expgroup_0.pt").write_bytes(b"fresh")

    monkeypatch.setattr("connito.shared.cycle.download_checkpoint_from_hf", fake_download_checkpoint_from_hf)

    hydrated = hydrate_miner_submissions_from_hf(
        config=config,
        subtensor=subtensor,
        validator_miner_assignment={"validator-hotkey": ["5DeoK1KCdM3a37ZY57BZMu6Y1psfQBBQ7buuho33czRMDBxT"]},
    )

    assert hydrated == 1
    assert len(downloaded) == 1
    assert not stale_file.exists()
    assert (tmp_path / "hotkey_5DeoK1KCdM3a37ZY57BZMu6Y1psfQBBQ7buuho33czRMDBxT_block_8040563.pt").exists()


def test_hydrate_miner_submissions_from_hf_prefers_hf_over_current_window_local_files(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    subtensor = SimpleNamespace(block=8040563)
    current_file = tmp_path / "hotkey_5DeoK1KCdM3a37ZY57BZMu6Y1psfQBBQ7buuho33czRMDBxT_block_8040555.pt"
    current_file.write_bytes(b"current")

    monkeypatch.setattr(
        "connito.shared.checkpoints.build_chain_checkpoints_from_previous_phase",
        lambda **kwargs: SimpleNamespace(checkpoints=[_make_chain_checkpoint("5DeoK1KCdM3a37ZY57BZMu6Y1psfQBBQ7buuho33czRMDBxT")]),
    )

    downloaded = []

    def fake_download_checkpoint_from_hf(**kwargs):
        downloaded.append(kwargs)
        (kwargs["dest_dir"] / "model_expgroup_0.pt").write_bytes(b"fresh")

    monkeypatch.setattr("connito.shared.cycle.download_checkpoint_from_hf", fake_download_checkpoint_from_hf)

    hydrated = hydrate_miner_submissions_from_hf(
        config=config,
        subtensor=subtensor,
        validator_miner_assignment={"validator-hotkey": ["5DeoK1KCdM3a37ZY57BZMu6Y1psfQBBQ7buuho33czRMDBxT"]},
    )

    assert hydrated == 1
    assert len(downloaded) == 1
    assert not current_file.exists()
    assert (tmp_path / "hotkey_5DeoK1KCdM3a37ZY57BZMu6Y1psfQBBQ7buuho33czRMDBxT_block_8040563.pt").exists()


def test_hydrate_miner_submissions_from_hf_keeps_local_file_when_hf_fails(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    subtensor = SimpleNamespace(block=8040563)
    current_file = tmp_path / "hotkey_5DeoK1KCdM3a37ZY57BZMu6Y1psfQBBQ7buuho33czRMDBxT_block_8040555.pt"
    current_file.write_bytes(b"current")

    monkeypatch.setattr(
        "connito.shared.checkpoints.build_chain_checkpoints_from_previous_phase",
        lambda **kwargs: SimpleNamespace(checkpoints=[_make_chain_checkpoint("5DeoK1KCdM3a37ZY57BZMu6Y1psfQBBQ7buuho33czRMDBxT")]),
    )
    monkeypatch.setattr(
        "connito.shared.cycle.download_checkpoint_from_hf",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("hf down")),
    )

    hydrated = hydrate_miner_submissions_from_hf(
        config=config,
        subtensor=subtensor,
        validator_miner_assignment={"validator-hotkey": ["5DeoK1KCdM3a37ZY57BZMu6Y1psfQBBQ7buuho33czRMDBxT"]},
    )

    assert hydrated == 0
    assert current_file.exists()
