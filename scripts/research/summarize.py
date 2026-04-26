from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.common import (
    RESEARCH_DIR,
    atomic_write_json,
    atomic_write_text,
    read_json,
    utc_now_iso,
)
from scripts.research.llm_clients import LLMError, generate_text, load_llm_config


SUMMARY_SCHEMA_HINT = {
    "tldr": "1-2 sentences",
    "problem_and_scope": "what the paper addresses",
    "key_methods": ["method bullet", "method bullet"],
    "decoding_and_lm": ["beam search, LM fusion, lexicon, rescoring, etc"],
    "post_processing": ["spell correction, normalization, error correction model, etc"],
    "training_tricks": ["augmentation, curriculum, distillation, etc"],
    "datasets_and_language": ["dataset names, language (Vietnamese?), handwritten vs printed"],
    "results": ["main metrics (CER/WER), compare to baselines"],
    "applicable_to_this_repo": ["concrete ideas to apply to VietOCR pipeline"],
    "implementation_notes": ["what modules/files likely impacted, rough approach"],
    "risks_and_costs": ["compute, dependency, licensing, failure modes"],
    "tags": ["lm_rescoring", "error_correction", "decoder", "vietnamesespecific"],
}


def _chunk_text(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # Prefer to split on paragraph boundary
        cut = text.rfind("\n\n", start, end)
        if cut == -1 or cut < start + int(max_chars * 0.5):
            cut = end
        chunks.append(text[start:cut].strip())
        start = cut
    return [c for c in chunks if c]


def _paper_prompt(meta: dict[str, Any], chunk: str, *, part: int, total_parts: int) -> str:
    title = meta.get("title") or ""
    year = meta.get("year") or ""
    venue = meta.get("venue") or ""
    authors = meta.get("authors") or ""

    return (
        "Summarize the following research paper content for an engineer improving a Vietnamese handwritten OCR pipeline.\n"
        "Return STRICT JSON only (no markdown, no backticks). Follow this schema exactly.\n\n"
        f"Schema example keys: {json.dumps(SUMMARY_SCHEMA_HINT, ensure_ascii=False)}\n\n"
        "Rules:\n"
        "- If a field is unknown from the text chunk, use empty string/empty list.\n"
        "- Be concrete and actionable for OCR recognition + post-processing.\n"
        "- Prefer short items; avoid filler.\n\n"
        f"Paper metadata:\nTitle: {title}\nYear: {year}\nVenue: {venue}\nAuthors: {authors}\n\n"
        f"Content chunk ({part}/{total_parts}):\n{chunk}\n"
    )


def _merge_partial_summaries(partials: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "tldr": "",
        "problem_and_scope": "",
        "key_methods": [],
        "decoding_and_lm": [],
        "post_processing": [],
        "training_tricks": [],
        "datasets_and_language": [],
        "results": [],
        "applicable_to_this_repo": [],
        "implementation_notes": [],
        "risks_and_costs": [],
        "tags": [],
    }

    def add_list(key: str) -> None:
        seen = set()
        out: list[str] = []
        for p in partials:
            vals = p.get(key) or []
            if not isinstance(vals, list):
                continue
            for v in vals:
                s = str(v).strip()
                if not s:
                    continue
                if s.lower() in seen:
                    continue
                seen.add(s.lower())
                out.append(s)
        merged[key] = out

    for k in ["key_methods", "decoding_and_lm", "post_processing", "training_tricks", "datasets_and_language", "results",
              "applicable_to_this_repo", "implementation_notes", "risks_and_costs", "tags"]:
        add_list(k)

    # Best-effort pick first non-empty strings for scalar fields
    for k in ["tldr", "problem_and_scope"]:
        for p in partials:
            val = (p.get(k) or "").strip()
            if val:
                merged[k] = val
                break

    return merged


def _render_md(meta: dict[str, Any], summary: dict[str, Any]) -> str:
    def section(title: str, items: list[str]) -> str:
        if not items:
            return f"## {title}\n\n- (none)\n\n"
        return "## " + title + "\n\n" + "\n".join(f"- {i}" for i in items) + "\n\n"

    title = meta.get("title") or ""
    year = meta.get("year") or ""
    venue = meta.get("venue") or ""
    authors = meta.get("authors") or ""
    doi = meta.get("doi") or ""
    arxiv = meta.get("arxiv_id") or ""
    s2 = meta.get("s2_url") or ""

    md = []
    md.append(f"# {title}\n\n")
    md.append(f"- Year: {year}\n")
    if venue:
        md.append(f"- Venue: {venue}\n")
    if authors:
        md.append(f"- Authors: {authors}\n")
    if doi:
        md.append(f"- DOI: `{doi}`\n")
    if arxiv:
        md.append(f"- arXiv: `{arxiv}`\n")
    if s2:
        md.append(f"- Link: {s2}\n")
    md.append("\n")
    md.append("## TL;DR\n\n")
    md.append((summary.get("tldr") or "(missing)") + "\n\n")
    md.append("## Problem & Scope\n\n")
    md.append((summary.get("problem_and_scope") or "(missing)") + "\n\n")

    md.append(section("Key Methods", summary.get("key_methods") or []))
    md.append(section("Decoding & LM", summary.get("decoding_and_lm") or []))
    md.append(section("Post-processing", summary.get("post_processing") or []))
    md.append(section("Training Tricks", summary.get("training_tricks") or []))
    md.append(section("Datasets & Language", summary.get("datasets_and_language") or []))
    md.append(section("Results", summary.get("results") or []))
    md.append(section("Applicable to This Repo", summary.get("applicable_to_this_repo") or []))
    md.append(section("Implementation Notes", summary.get("implementation_notes") or []))
    md.append(section("Risks & Costs", summary.get("risks_and_costs") or []))
    md.append(section("Tags", summary.get("tags") or []))
    return "".join(md)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize extracted paper text with LLM into research/summaries/.")
    ap.add_argument("--manifest", type=str, default=str(RESEARCH_DIR / "manifest.json"))
    ap.add_argument("--max-papers", type=int, default=0, help="0 = no limit; useful for smoke tests.")
    ap.add_argument("--max-chars-per-chunk", type=int, default=12000)
    ap.add_argument("--max-input-chars", type=int, default=60000, help="Cap input length per paper (0 = no cap).")
    ap.add_argument("--force", action="store_true", help="Resummarize even if outputs exist.")
    args = ap.parse_args()

    cfg = load_llm_config()
    print(f"LLM provider={cfg.provider} model={cfg.model}")

    manifest_path = Path(args.manifest)
    manifest = read_json(manifest_path)
    papers: list[dict[str, Any]] = list(manifest.get("papers") or [])

    text_dir = RESEARCH_DIR / "text"
    out_dir = RESEARCH_DIR / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    summarized = 0
    for paper in papers:
        tx = paper.get("text") or {}
        if tx.get("status") != "extracted":
            continue
        paper_id = paper.get("paper_id")
        txt_path = Path((tx.get("txt_path") or str(text_dir / f"{paper_id}.txt"))).resolve()
        if not txt_path.exists():
            continue

        md_path = out_dir / f"{paper_id}.md"
        js_path = out_dir / f"{paper_id}.json"
        sm = paper.get("summary") or {}
        if not args.force and md_path.exists() and js_path.exists():
            sm.update({"status": "summarized", "md_path": str(md_path), "json_path": str(js_path), "error": ""})
            paper["summary"] = sm
            continue

        try:
            raw_text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if args.max_input_chars and args.max_input_chars > 0:
                raw_text = raw_text[: int(args.max_input_chars)]
            chunks = _chunk_text(raw_text, max_chars=int(args.max_chars_per_chunk))
            partials: list[dict[str, Any]] = []
            for idx, chunk in enumerate(chunks, start=1):
                prompt = _paper_prompt(paper, chunk, part=idx, total_parts=len(chunks))
                out = generate_text(prompt)
                try:
                    partial = json.loads(out)
                except Exception:
                    partial = {"tldr": "", "problem_and_scope": "", "key_methods": [], "decoding_and_lm": [], "post_processing": [],
                               "training_tricks": [], "datasets_and_language": [], "results": [], "applicable_to_this_repo": [],
                               "implementation_notes": [], "risks_and_costs": [], "tags": [], "parse_error_raw": out}
                partials.append(partial)

            merged = _merge_partial_summaries(partials)
            atomic_write_json(
                js_path,
                {
                    "paper_id": paper_id,
                    "summarized_at": utc_now_iso(),
                    "llm_provider": cfg.provider,
                    "llm_model": cfg.model,
                    "summary": merged,
                },
            )
            atomic_write_text(md_path, _render_md(paper, merged))

            sm.update({"status": "summarized", "md_path": str(md_path), "json_path": str(js_path), "error": ""})
            paper["summary"] = sm
        except LLMError as e:
            sm.update({"status": "error", "md_path": "", "json_path": "", "error": str(e)})
            paper["summary"] = sm
        except Exception as e:
            sm.update({"status": "error", "md_path": "", "json_path": "", "error": str(e)})
            paper["summary"] = sm

        summarized += 1
        if args.max_papers and summarized >= int(args.max_papers):
            break

    # Build index.md
    index_md = out_dir / "index.md"
    lines = ["# Paper Summaries\n\n", f"_Generated at {utc_now_iso()}_\n\n"]
    for paper in papers:
        sm = paper.get("summary") or {}
        if sm.get("status") != "summarized":
            continue
        paper_id = paper.get("paper_id")
        title = paper.get("title") or ""
        year = paper.get("year") or ""
        rel = f"{paper_id}.md"
        lines.append(f"- [{title} ({year})]({rel})\n")
    atomic_write_text(index_md, "".join(lines))

    manifest["updated_at"] = utc_now_iso()
    manifest["papers"] = papers
    atomic_write_json(manifest_path, manifest)

    print(f"Updated manifest: {manifest_path}")
    print(f"Summaries:        {out_dir}")


if __name__ == "__main__":
    main()
