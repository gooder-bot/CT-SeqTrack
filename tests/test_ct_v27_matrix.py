"""Reviewable commands, exact run matrix, and no accidental training in plan export."""
import json
from pathlib import Path
import subprocess

import pytest

from tools.run_ct_v27_matrix import (
    ARMS, CATEGORIES, ROOT, build_matrix, execute_training, main,
)


@pytest.mark.parametrize("stage,count,categories", (("mini", 6, ("Car",)), ("full", 30, CATEGORIES)))
def test_matrix_has_six_arms_per_category_with_per_checkpoint_full_calibration(tmp_path, stage, count, categories):
    matrix, configs = build_matrix(stage, "/server/nuscenes", tmp_path / stage)
    assert matrix["run_count"] == len(matrix["runs"]) == len(configs) == count
    assert set(matrix["categories"]) == set(categories)
    assert {(run["category"], run["arm"]) for run in matrix["runs"]} == {
        (category, arm) for category in categories for arm in ARMS}
    for run in matrix["runs"]:
        assert configs[run["resolved_config"]]["category_name"] == run["category"]
        assert configs[run["resolved_config"]]["batch_size"] == 16
        assert "--checkpoint" not in run["train_argv"] and "--init_checkpoint" not in run["train_argv"]
        assert [task["epoch"] for task in run["next_commands"]] == [58, 59, 60]
        for task in run["next_commands"]:
            assert f"epoch={task['epoch']:03d}.ckpt" in task["checkpoint"]
            assert ("calibration_argv" in task) == (run["arm"] == "full")
            if run["arm"] == "full":
                assert task["calibration_argv"][task["calibration_argv"].index("--config") + 1] == run["resolved_config"]
                assert task["evaluation_argv"][-1] == task["calibration_artifact"]
    assert matrix["reporting"]["final_epoch"] == 60
    assert matrix["reporting"]["late3_epochs"] == [58, 59, 60]


def test_default_export_never_launches_training_or_evaluation(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("review export must not launch a subprocess")
    monkeypatch.setattr(subprocess, "run", unexpected)
    directory = tmp_path / "review"
    main(["--stage", "mini", "--path", "/data/not_required_during_review", "--output", str(directory)])
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert not manifest["execution_requested"] and not manifest["evaluation_executed"]
    assert {run["training_status"] for run in manifest["runs"]} == {"not_started"}
    assert (directory / "commands.sh").is_file() and (directory / "commands.ps1").is_file()
    assert not (directory / "runs").exists()


def test_protected_outputs_are_rejected_before_any_write():
    with pytest.raises(ValueError, match="protected"):
        build_matrix("mini", "/data", ROOT / "output" / "must_not_exist_v27_matrix")


def test_execute_stops_on_first_failure_and_does_not_start_pending_evaluations(tmp_path, monkeypatch):
    matrix, _ = build_matrix("mini", str(tmp_path), tmp_path)
    calls = []
    def fail_first(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode=2)
    monkeypatch.setattr(subprocess, "run", fail_first)
    with pytest.raises(subprocess.CalledProcessError):
        execute_training(matrix, tmp_path)
    assert calls == [matrix["runs"][0]["train_argv"]]
    assert matrix["runs"][0]["training_status"] == "failed"
    assert all(run["training_status"] == "not_started" for run in matrix["runs"][1:])
    assert not matrix["evaluation_executed"]


def test_execute_refuses_existing_run_content_before_starting_any_training(tmp_path, monkeypatch):
    matrix, _ = build_matrix("mini", str(tmp_path), tmp_path)
    existing = Path(matrix["runs"][-1]["run_directory"])
    existing.mkdir(parents=True)
    (existing / "checkpoint_placeholder").write_text("existing run", encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("must fail before training"))
    with pytest.raises(ValueError, match="nonempty"):
        execute_training(matrix, tmp_path)
