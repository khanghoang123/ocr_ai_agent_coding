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


def extract_pdf_text(pdf_path: Path) -> dict[str, Any]:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError("PyMuPDF is required. Install with: pip install pymupdf") from e

    doc = fitz.open(pdf_path)
    pages: list[dict[str, Any]] = []
    full_text_parts: list[str] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        pages.append({"page": i + 1, "text": text})
        if text.strip():
            full_text_parts.append(text)
    doc.close()

    full_text = "\n\n".join(full_text_parts).strip()
    return {"pages": pages, "full_text": full_text}


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract text from downloaded PDFs into research/text/.")
    ap.add_argument("--manifest", type=str, default=str(RESEARCH_DIR / "manifest.json"))
    ap.add_argument("--max-papers", type=int, default=0, help="0 = no limit; useful for smoke tests.")
    ap.add_argument("--force", action="store_true", help="Re-extract even if outputs exist.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    manifest = read_json(manifest_path)
    papers: list[dict[str, Any]] = list(manifest.get("papers") or [])

    out_dir = RESEARCH_DIR / "text"
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for paper in papers:
        dl = paper.get("download") or {}
        if dl.get("status") != "downloaded":
            continue
        pdf_path_str = (dl.get("pdf_path") or "").strip()
        if not pdf_path_str:
            continue
        pdf_path = Path(pdf_path_str)
        if not pdf_path.exists():
            continue

        paper_id = paper.get("paper_id")
        txt_path = out_dir / f"{paper_id}.txt"
        json_path = out_dir / f"{paper_id}.json"

        tx = paper.get("text") or {}
        if not args.force and txt_path.exists() and json_path.exists():
            tx.update({"status": "extracted", "txt_path": str(txt_path), "json_path": str(json_path), "error": ""})
            paper["text"] = tx
            continue

        try:
            extracted = extract_pdf_text(pdf_path)
            atomic_write_text(txt_path, extracted["full_text"] + "\n")
            atomic_write_json(
                json_path,
                {
                    "paper_id": paper_id,
                    "pdf_path": str(pdf_path),
                    "extracted_at": utc_now_iso(),
                    "pages": extracted["pages"],
                },
            )
            tx.update({"status": "extracted", "txt_path": str(txt_path), "json_path": str(json_path), "error": ""})
            paper["text"] = tx
        except Exception as e:
            tx.update({"status": "error", "txt_path": "", "json_path": "", "error": str(e)})
            paper["text"] = tx

        processed += 1
        if args.max_papers and processed >= int(args.max_papers):
            break

    manifest["updated_at"] = utc_now_iso()
    manifest["papers"] = papers
    atomic_write_json(manifest_path, manifest)

    print(f"Updated manifest: {manifest_path}")
    print(f"Text folder:      {out_dir}")


if __name__ == "__main__":
    main()
