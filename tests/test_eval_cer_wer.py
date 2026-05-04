"""Unit tests for ``scripts/run_eval_cer_wer.py``."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_eval_cer_wer.py"


@pytest.fixture(scope="module")
def eval_module():
    """Load ``scripts/run_eval_cer_wer.py`` as a module so tests can call its helpers."""
    spec = importlib.util.spec_from_file_location("run_eval_cer_wer", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_eval_cer_wer"] = module
    spec.loader.exec_module(module)
    return module


def test_strip_image_id_prefix(eval_module):
    assert eval_module._strip_image_id_prefix("0006_thumb_1200_1698") == "thumb_1200_1698"
    assert eval_module._strip_image_id_prefix("12_essay_sample_page") == "essay_sample_page"
    # No prefix -> identity.
    assert eval_module._strip_image_id_prefix("plain_name") == "plain_name"


def test_load_ground_truth(eval_module, tmp_path: Path):
    payload = {
        "image": "page1.png",
        "ground_truth_quality": "verified",
        "notes": "test",
        "lines": [{"text": "Xin chào"}, {"text": "thế giới"}],
    }
    (tmp_path / "page1.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    gt = eval_module.load_ground_truth(tmp_path)
    assert "page1" in gt
    assert gt["page1"].quality == "verified"
    assert [line.text for line in gt["page1"].lines] == ["Xin chào", "thế giới"]


def test_line_level_perfect_match(eval_module):
    pred = ["Xin chào", "thế giới"]
    ref = ["Xin chào", "thế giới"]
    metrics = eval_module._line_level_cer_wer(pred, ref)
    assert metrics["matched_pairs"] == 2
    assert metrics["unmatched_pred"] == 0
    assert metrics["unmatched_ref"] == 0
    assert metrics["line_cer"] == pytest.approx(0.0)
    assert metrics["line_wer"] == pytest.approx(0.0)


def test_line_level_handles_reordering(eval_module):
    """Reading-order doesn't matter — Hungarian matching pairs equal lines."""
    pred = ["thế giới", "Xin chào"]
    ref = ["Xin chào", "thế giới"]
    metrics = eval_module._line_level_cer_wer(pred, ref)
    assert metrics["matched_pairs"] == 2
    assert metrics["line_cer"] == pytest.approx(0.0)


def test_line_level_partial_match_with_unmatched(eval_module):
    pred = ["Xin chào", "noise here"]
    ref = ["Xin chào", "tạm biệt", "Hello"]
    metrics = eval_module._line_level_cer_wer(pred, ref)
    # 2 predictions matched against 2 of 3 references, 1 reference unmatched.
    assert metrics["matched_pairs"] == 2
    assert metrics["unmatched_pred"] == 0
    assert metrics["unmatched_ref"] == 1
    # line_cer = (0 + something + 1*1) / 3 -> bounded > 0.33.
    assert metrics["line_cer"] > 0.0
    assert metrics["line_cer"] < 1.0


def test_line_level_all_unmatched_pred(eval_module):
    """Predictions present, no references: every prediction is a false positive."""
    metrics = eval_module._line_level_cer_wer(["something"], [])
    assert metrics["matched_pairs"] == 0
    assert metrics["unmatched_pred"] == 1
    assert metrics["line_cer"] == pytest.approx(1.0)


def test_line_level_all_unmatched_ref(eval_module):
    """References present, no predictions: every reference is a missed line."""
    metrics = eval_module._line_level_cer_wer([], ["Xin chào"])
    assert metrics["matched_pairs"] == 0
    assert metrics["unmatched_ref"] == 1
    assert metrics["line_cer"] == pytest.approx(1.0)


def test_evaluate_run_skips_unknown_images(eval_module, tmp_path: Path):
    """Images present in the run but missing from GT are silently skipped."""
    run_dir = tmp_path / "run"
    exp_dir = run_dir / "E0_test"
    debug_dir = exp_dir / "debug"
    (debug_dir / "0000_known_page").mkdir(parents=True)
    (debug_dir / "0001_unknown_page").mkdir(parents=True)
    (debug_dir / "0000_known_page" / "text.txt").write_text("Xin chào", encoding="utf-8")
    (debug_dir / "0001_unknown_page" / "text.txt").write_text("ignored", encoding="utf-8")

    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "known_page.json").write_text(
        json.dumps({"image": "known_page.png", "ground_truth_quality": "verified",
                    "lines": [{"text": "Xin chào"}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    gt = eval_module.load_ground_truth(gt_dir)
    report = eval_module.evaluate_run(run_dir, gt)
    assert "E0_test" in report["experiments"]
    per_image = report["experiments"]["E0_test"]["per_image"]
    assert len(per_image) == 1
    assert per_image[0]["image_stem"] == "known_page"
    assert per_image[0]["page_cer"] == pytest.approx(0.0)


def test_evaluate_run_handles_empty_predictions(eval_module, tmp_path: Path):
    """A backend that produced no predictions for an image still appears in the report."""
    run_dir = tmp_path / "run"
    debug_dir = run_dir / "E1_empty" / "debug" / "0000_pg"
    debug_dir.mkdir(parents=True)
    (debug_dir / "text.txt").write_text("\n\n", encoding="utf-8")
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "pg.json").write_text(
        json.dumps({"image": "pg.png", "ground_truth_quality": "verified",
                    "lines": [{"text": "abc"}, {"text": "def"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    gt = eval_module.load_ground_truth(gt_dir)
    report = eval_module.evaluate_run(run_dir, gt)
    pg = report["experiments"]["E1_empty"]["per_image"][0]
    assert pg["predicted_lines"] == 0
    assert pg["reference_lines"] == 2
    # No predictions -> page CER >= 1 (need to insert all of ref).
    assert pg["page_cer"] >= 1.0
    # Line-level: every reference line is unmatched.
    assert pg["unmatched_ref"] == 2
    assert pg["matched_pairs"] == 0
