from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.common import (
    RESEARCH_DIR,
    atomic_write_json,
    compute_paper_id,
    http_download_file,
    http_get_json,
    read_json,
    slugify,
    utc_now_iso,
)


OPENALEX_DOI_URL = "https://api.openalex.org/works/doi:{doi}"


def _pick_best_pdf_url(paper: dict[str, Any]) -> tuple[str, str]:
    """
    Returns (pdf_url, source).
    Sources: semantic_scholar | arxiv | openalex
    """
    oa = (paper.get("open_access_pdf_url") or "").strip()
    if oa:
        return oa, "semantic_scholar"

    arxiv = (paper.get("arxiv_id") or "").strip()
    if arxiv:
        return f"https://arxiv.org/pdf/{arxiv}.pdf", "arxiv"

    doi = (paper.get("doi") or "").strip()
    if doi:
        try:
            work = http_get_json(OPENALEX_DOI_URL.format(doi=doi))
            boa = work.get("best_oa_location") or {}
            pdf_url = (boa.get("pdf_url") or "").strip()
            if pdf_url:
                if pdf_url.endswith(".pdf"):
                    return pdf_url, "openalex"
                # Some OA locations return arXiv without .pdf suffix
                if "arxiv.org/pdf/" in pdf_url:
                    return pdf_url if pdf_url.endswith(".pdf") else pdf_url + ".pdf", "openalex"
        except Exception:
            pass

    return "", ""


def _safe_pdf_filename(paper: dict[str, Any]) -> str:
    year = paper.get("year") or "unknown"
    title = paper.get("title") or "paper"
    paper_id = paper.get("paper_id") or compute_paper_id(
        doi=(paper.get("doi") or None),
        arxiv_id=(paper.get("arxiv_id") or None),
        title=str(title),
        year=int(year) if str(year).isdigit() else None,
    )
    return f"{year}_{slugify(str(title), max_len=70)}_{paper_id}.pdf"


def _update_missing_md(missing_md: Path, papers: list[dict[str, Any]]) -> None:
    lines = ["# Missing PDFs (legal OA/preprint only)\n\n"]
    lines.append(f"_Generated at {utc_now_iso()}_\n\n")
    lines.append(
        "These papers could not be downloaded from Semantic Scholar OA, arXiv, or OpenAlex OA locations.\n"
        "Suggestions: try searching the exact title + `pdf` or `preprint`, or the DOI on the author's homepage/institutional repo.\n\n"
    )

    missing = [p for p in papers if (p.get("download") or {}).get("status") == "missing"]
    if not missing:
        lines.append("✅ None missing.\n")
        missing_md.write_text("".join(lines), encoding="utf-8")
        return

    for p in missing:
        title = p.get("title") or ""
        year = p.get("year") or ""
        doi = p.get("doi") or ""
        arxiv = p.get("arxiv_id") or ""
        s2 = p.get("s2_url") or ""
        lines.append(f"## {title} ({year})\n\n")
        if doi:
            lines.append(f"- DOI: `{doi}`\n")
        if arxiv:
            lines.append(f"- arXiv: `{arxiv}`\n")
        if s2:
            lines.append(f"- Semantic Scholar: {s2}\n")
        lines.append("- Search hints:\n")
        lines.append(f"  - `{title}` download\n")
        if doi:
            lines.append(f"  - `{doi}` pdf\n")
        lines.append("\n")

    missing_md.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve OA/preprint PDF URLs and download PDFs into research/papers/.")
    ap.add_argument("--manifest", type=str, default=str(RESEARCH_DIR / "manifest.json"))
    ap.add_argument("--max-downloads", type=int, default=0, help="0 = no limit; useful for smoke tests.")
    ap.add_argument("--force", action="store_true", help="Redownload even if pdf_path exists.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = read_json(manifest_path)
    papers: list[dict[str, Any]] = list(manifest.get("papers") or [])

    out_dir = RESEARCH_DIR / "papers"
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for paper in papers:
        dl = paper.get("download") or {}
        existing_path = (dl.get("pdf_path") or "").strip()
        if existing_path and not args.force and Path(existing_path).exists():
            dl["status"] = "downloaded"
            paper["download"] = dl
            continue

        pdf_url, source = _pick_best_pdf_url(paper)
        if not pdf_url:
            dl.update({"status": "missing", "pdf_url": "", "pdf_source": "", "pdf_path": "", "error": "no_oa_pdf_found"})
            paper["download"] = dl
            continue

        filename = _safe_pdf_filename(paper)
        out_path = out_dir / filename
        try:
            http_download_file(pdf_url, out_path=out_path, max_bytes=200 * 1024 * 1024)
            dl.update(
                {
                    "status": "downloaded",
                    "pdf_url": pdf_url,
                    "pdf_source": source,
                    "pdf_path": str(out_path),
                    "error": "",
                }
            )
            paper["download"] = dl
            downloaded += 1
            if args.max_downloads and downloaded >= int(args.max_downloads):
                break
        except Exception as e:
            dl.update(
                {
                    "status": "error",
                    "pdf_url": pdf_url,
                    "pdf_source": source,
                    "pdf_path": "",
                    "error": str(e),
                }
            )
            paper["download"] = dl

    manifest["updated_at"] = utc_now_iso()
    manifest["papers"] = papers
    atomic_write_json(manifest_path, manifest)

    missing_md = manifest_path.parent / "missing_pdfs.md"
    _update_missing_md(missing_md, papers)

    print(f"Updated manifest: {manifest_path}")
    print(f"PDF folder:       {out_dir}")
    print(f"Missing list:     {missing_md}")


if __name__ == "__main__":
    main()
