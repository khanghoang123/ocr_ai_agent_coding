from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.common import (
    RESEARCH_DIR,
    compute_paper_id,
    normalize_title,
    atomic_write_json,
    utc_now_iso,
    write_csv,
    http_get_json,
)


S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = ",".join(
    [
        "paperId",
        "title",
        "year",
        "venue",
        "url",
        "abstract",
        "authors",
        "externalIds",
        "openAccessPdf",
        "isOpenAccess",
        "citationCount",
    ]
)


DEFAULT_QUERIES = [
    "vietnamese handwritten OCR post-processing",
    "handwritten text recognition language model rescoring",
    "OCR error correction sequence-to-sequence",
    "lexicon constrained decoding OCR beam search",
    "handwritten text recognition transformer decoder language model",
    "vietocr handwritten recognition",
]


def _authors_to_str(authors: Any) -> str:
    if not isinstance(authors, list):
        return ""
    names: list[str] = []
    for a in authors:
        if isinstance(a, dict) and a.get("name"):
            names.append(str(a["name"]))
    return "; ".join(names)


def _pick_external_id(external_ids: dict[str, Any] | None, key: str) -> Optional[str]:
    if not external_ids or not isinstance(external_ids, dict):
        return None
    val = external_ids.get(key)
    if val is None:
        return None
    return str(val).strip() or None


def _paper_from_s2(item: dict[str, Any], query: str) -> dict[str, Any]:
    title = (item.get("title") or "").strip()
    year = item.get("year")
    external = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
    doi = _pick_external_id(external, "DOI")
    arxiv_id = _pick_external_id(external, "ArXiv")
    s2_paper_id = item.get("paperId")

    paper_id = compute_paper_id(doi=doi, arxiv_id=arxiv_id, title=title, year=year)

    open_access_pdf_url = None
    oap = item.get("openAccessPdf")
    if isinstance(oap, dict):
        open_access_pdf_url = oap.get("url") or None

    return {
        "paper_id": paper_id,
        "title": title,
        "title_norm": normalize_title(title),
        "year": year,
        "venue": item.get("venue") or "",
        "authors": _authors_to_str(item.get("authors")),
        "doi": doi or "",
        "arxiv_id": arxiv_id or "",
        "s2_paper_id": s2_paper_id or "",
        "s2_url": item.get("url") or "",
        "abstract": (item.get("abstract") or "").strip(),
        "citation_count": int(item.get("citationCount") or 0),
        "is_open_access": bool(item.get("isOpenAccess") or False),
        "open_access_pdf_url": open_access_pdf_url or "",
        "queries": [query],
        "download": {
            "status": "pending",
            "pdf_url": "",
            "pdf_source": "",
            "pdf_path": "",
            "error": "",
        },
        "text": {"status": "pending", "txt_path": "", "json_path": "", "error": ""},
        "summary": {"status": "pending", "md_path": "", "json_path": "", "error": ""},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect related papers via Semantic Scholar (2021–2026 by default).")
    ap.add_argument("--query", action="append", default=[], help="Query string (repeatable).")
    ap.add_argument("--queries-file", type=str, default="", help="Text file with one query per line.")
    ap.add_argument("--year-from", type=int, default=2021)
    ap.add_argument("--year-to", type=int, default=2026)
    ap.add_argument("--max-papers", type=int, default=30, help="Max unique papers in manifest.")
    ap.add_argument("--per-query-limit", type=int, default=25, help="How many results to pull per query (pre-dedup).")
    ap.add_argument("--sleep-s", type=float, default=1.0, help="Sleep between API calls to reduce 429s.")
    ap.add_argument("--out", type=str, default=str(RESEARCH_DIR / "manifest.json"))
    args = ap.parse_args()

    queries: list[str] = list(args.query)
    if args.queries_file:
        qpath = Path(args.queries_file)
        for line in qpath.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    if not queries:
        queries = DEFAULT_QUERIES

    year_from = int(args.year_from)
    year_to = int(args.year_to)
    max_papers = int(args.max_papers)

    papers_by_key: dict[str, dict[str, Any]] = {}
    for q in queries:
        offset = 0
        pulled = 0
        while pulled < args.per_query_limit:
            batch_limit = min(100, args.per_query_limit - pulled)
            try:
                data = http_get_json(
                    S2_SEARCH_URL,
                    params={
                        "query": q,
                        "limit": batch_limit,
                        "offset": offset,
                        "fields": S2_FIELDS,
                    },
                )
            except Exception as e:
                print(f"[warn] Semantic Scholar query failed for '{q}': {e}")
                break
            items = data.get("data") or []
            if not items:
                break
            for item in items:
                year = item.get("year")
                if year is None or not (year_from <= int(year) <= year_to):
                    continue
                paper = _paper_from_s2(item, q)
                dedup_key = paper["doi"] or paper["arxiv_id"] or paper["title_norm"]
                existing = papers_by_key.get(dedup_key)
                if existing is None:
                    papers_by_key[dedup_key] = paper
                else:
                    # Merge queries + keep higher citation_count / richer abstract
                    existing["queries"] = sorted(set(existing.get("queries", []) + [q]))
                    if int(paper.get("citation_count") or 0) > int(existing.get("citation_count") or 0):
                        existing["citation_count"] = paper["citation_count"]
                    if not existing.get("abstract") and paper.get("abstract"):
                        existing["abstract"] = paper["abstract"]
                    if not existing.get("open_access_pdf_url") and paper.get("open_access_pdf_url"):
                        existing["open_access_pdf_url"] = paper["open_access_pdf_url"]
            pulled += len(items)
            offset = int(data.get("next") or (offset + len(items)))
            time.sleep(max(0.0, float(args.sleep_s)))
            if pulled >= args.per_query_limit:
                break

    papers = list(papers_by_key.values())
    papers.sort(key=lambda p: (int(p.get("citation_count") or 0), int(p.get("year") or 0)), reverse=True)
    papers = papers[:max_papers]

    manifest = {
        "generated_at": utc_now_iso(),
        "config": {
            "year_from": year_from,
            "year_to": year_to,
            "max_papers": max_papers,
            "queries": queries,
            "source": "semantic_scholar",
        },
        "papers": papers,
    }

    out_path = Path(args.out)
    atomic_write_json(out_path, manifest)

    csv_path = out_path.parent / "papers.csv"
    write_csv(
        csv_path,
        papers,
        fieldnames=[
            "paper_id",
            "year",
            "title",
            "authors",
            "venue",
            "doi",
            "arxiv_id",
            "citation_count",
            "is_open_access",
            "open_access_pdf_url",
            "s2_url",
        ],
    )

    missing_md = out_path.parent / "missing_pdfs.md"
    missing_lines = [
        "# Missing PDFs (legal OA/preprint only)\n",
        "\n",
        "This file is generated after download attempts. Run `resolve_and_download.py` first.\n",
        "\n",
    ]
    if not missing_md.exists():
        missing_md.write_text("".join(missing_lines), encoding="utf-8")

    print(f"Wrote manifest: {out_path}")
    print(f"Wrote CSV:      {csv_path}")


if __name__ == "__main__":
    main()
