from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.common import RESEARCH_DIR, atomic_write_text, read_json, utc_now_iso


TAG_BUCKETS = {
    "lm_rescoring": "A) Decoder / LM Rescoring",
    "decoder": "A) Decoder / LM Rescoring",
    "beam_search": "A) Decoder / LM Rescoring",
    "error_correction": "B) Vietnamese Error-Correction",
    "spelling": "B) Vietnamese Error-Correction",
    "noisy_channel": "B) Vietnamese Error-Correction",
    "augmentation": "C) Training Tricks",
    "distillation": "C) Training Tricks",
    "curriculum": "C) Training Tricks",
    "evaluation": "D) Evaluation & Ablation",
    "ablation": "D) Evaluation & Ablation",
}


def _load_summaries(summaries_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for js in sorted(summaries_dir.glob("*.json")):
        if js.name == "index.json":
            continue
        try:
            obj = json.loads(js.read_text(encoding="utf-8"))
            items.append(obj)
        except Exception:
            continue
    return items


def _bucket_for(tags: list[str]) -> str:
    for t in tags:
        key = t.strip().lower()
        if key in TAG_BUCKETS:
            return TAG_BUCKETS[key]
    return "Other"


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesize all paper summaries into a concrete improvement checklist.")
    ap.add_argument("--manifest", type=str, default=str(RESEARCH_DIR / "manifest.json"))
    ap.add_argument("--out", type=str, default=str(RESEARCH_DIR / "improvements.md"))
    args = ap.parse_args()

    manifest = read_json(Path(args.manifest))
    papers_by_id = {p.get("paper_id"): p for p in (manifest.get("papers") or [])}

    summaries_dir = RESEARCH_DIR / "summaries"
    summaries = _load_summaries(summaries_dir)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in summaries:
        paper_id = s.get("paper_id")
        paper = papers_by_id.get(paper_id, {})
        summary = (s.get("summary") or {}) if isinstance(s.get("summary"), dict) else {}
        tags = summary.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        bucket = _bucket_for([str(t) for t in tags])
        buckets[bucket].append({"paper": paper, "summary": summary})

    def render_paper_ref(paper: dict[str, Any]) -> str:
        title = paper.get("title") or ""
        year = paper.get("year") or ""
        doi = paper.get("doi") or ""
        arxiv = paper.get("arxiv_id") or ""
        parts = [f"{title} ({year})"]
        if doi:
            parts.append(f"DOI {doi}")
        if arxiv:
            parts.append(f"arXiv {arxiv}")
        return " — ".join(parts)

    out_lines: list[str] = []
    out_lines.append("# Improvements from 2021–2026 Paper Review\n\n")
    out_lines.append(f"_Generated at {utc_now_iso()}_\n\n")
    out_lines.append("This document aggregates actionable methods to improve the repo's OCR recognition + post-processing.\n")
    out_lines.append("It is derived from `research/summaries/*.json` and mapped to likely implementation areas such as `src/ocr_pipeline/post_processing.py`.\n\n")

    ordered_sections = [
        "A) Decoder / LM Rescoring",
        "B) Vietnamese Error-Correction",
        "C) Training Tricks",
        "D) Evaluation & Ablation",
        "Other",
    ]
    for section in ordered_sections:
        items = buckets.get(section, [])
        out_lines.append(f"## {section}\n\n")
        if not items:
            out_lines.append("- (none)\n\n")
            continue

        # Build quick wins / medium / long-term heuristic by counting implementation_notes + risks
        for it in items:
            paper = it["paper"]
            summ = it["summary"]
            out_lines.append(f"### {render_paper_ref(paper)}\n\n")
            tldr = (summ.get("tldr") or "").strip()
            if tldr:
                out_lines.append(f"- TL;DR: {tldr}\n")
            appl = summ.get("applicable_to_this_repo") or []
            if isinstance(appl, list) and appl:
                out_lines.append("- Ideas to apply:\n")
                for a in appl[:8]:
                    out_lines.append(f"  - {str(a).strip()}\n")
            notes = summ.get("implementation_notes") or []
            if isinstance(notes, list) and notes:
                out_lines.append("- Implementation notes:\n")
                for n in notes[:6]:
                    out_lines.append(f"  - {str(n).strip()}\n")
            risks = summ.get("risks_and_costs") or []
            if isinstance(risks, list) and risks:
                out_lines.append("- Risks/costs:\n")
                for r in risks[:6]:
                    out_lines.append(f"  - {str(r).strip()}\n")
            out_lines.append("\n")

    atomic_write_text(Path(args.out), "".join(out_lines))
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
