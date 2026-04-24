from pathlib import Path

from connito.shared.checkpoints import build_local_checkpoints, delete_old_checkpoints


def test_build_local_checkpoints_ignores_score_aggregator_sidecar(tmp_path):
    (tmp_path / "globalver_10").mkdir()
    (tmp_path / "score_aggregator.json").write_text("{}", encoding="utf-8")

    checkpoints = build_local_checkpoints(tmp_path).ordered()

    assert [checkpoint.path.name for checkpoint in checkpoints] == ["globalver_10"]


def test_delete_old_checkpoints_preserves_score_aggregator_sidecar(tmp_path):
    old_ckpt = tmp_path / "globalver_10"
    new_ckpt = tmp_path / "globalver_20"
    old_ckpt.mkdir()
    new_ckpt.mkdir()
    score_path = tmp_path / "score_aggregator.json"
    score_path.write_text("{}", encoding="utf-8")

    removed = delete_old_checkpoints(tmp_path, topk=1)

    assert [Path(path).name for path in removed] == ["globalver_10"]
    assert new_ckpt.exists()
    assert score_path.exists()
