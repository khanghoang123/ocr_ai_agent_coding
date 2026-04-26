from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import requests


ROOT_DIR = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT_DIR / "research"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        return "paper"
    return text[:max_len].rstrip("-")


def normalize_title(title: str) -> str:
    title = title.strip().lower()
    title = re.sub(r"\s+", " ", title)
    title = re.sub(r"[^a-z0-9 ]+", "", title)
    return title.strip()


def compute_paper_id(*, doi: Optional[str], arxiv_id: Optional[str], title: str, year: Optional[int]) -> str:
    key = ""
    if doi:
        key = f"doi:{doi.strip().lower()}"
    elif arxiv_id:
        key = f"arxiv:{arxiv_id.strip().lower()}"
    else:
        key = f"title:{normalize_title(title)}|year:{year or ''}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class HttpConfig:
    timeout_s: int = 45
    max_retries: int = 5
    backoff_s: float = 1.0
    user_agent: str = "ocr-ai-agent-research/1.0 (+https://local)"


def http_get_json(url: str, *, params: Optional[dict[str, Any]] = None, cfg: Optional[HttpConfig] = None) -> Any:
    cfg = cfg or HttpConfig()
    sess = requests.Session()
    headers = {"User-Agent": cfg.user_agent}
    # Optional API key support (helps with strict rate limits).
    # Semantic Scholar: https://www.semanticscholar.org/product/api
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if s2_key and "semanticscholar.org" in url:
        headers["x-api-key"] = s2_key.strip()
    last_err: Optional[Exception] = None
    for attempt in range(cfg.max_retries):
        try:
            r = sess.get(url, params=params, headers=headers, timeout=cfg.timeout_s)
            if r.status_code in (429, 500, 502, 503, 504):
                if r.status_code == 429:
                    retry_after = r.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        time.sleep(int(retry_after))
                    else:
                        time.sleep(max(10.0, cfg.backoff_s * (2**attempt)))
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(cfg.backoff_s * (2**attempt))
    raise RuntimeError(f"Failed GET JSON after retries: {url}") from last_err


def http_download_file(
    url: str,
    *,
    out_path: Path,
    cfg: Optional[HttpConfig] = None,
    max_bytes: Optional[int] = None,
) -> None:
    cfg = cfg or HttpConfig(timeout_s=90)
    sess = requests.Session()
    headers = {"User-Agent": cfg.user_agent}
    ensure_dir(out_path.parent)
    tmp = out_path.with_suffix(out_path.suffix + ".part")

    last_err: Optional[Exception] = None
    for attempt in range(cfg.max_retries):
        try:
            with sess.get(url, headers=headers, timeout=cfg.timeout_s, stream=True, allow_redirects=True) as r:
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()

                total = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 128):
                        if not chunk:
                            continue
                        f.write(chunk)
                        total += len(chunk)
                        if max_bytes and total > max_bytes:
                            raise RuntimeError(f"Download too large (> {max_bytes} bytes)")

            # Validate PDF magic header
            head = tmp.read_bytes()[:8]
            if not head.startswith(b"%PDF"):
                raise RuntimeError("Downloaded file is not a PDF (missing %PDF header)")

            tmp.replace(out_path)
            return
        except Exception as e:
            last_err = e
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            time.sleep(cfg.backoff_s * (2**attempt))
    raise RuntimeError(f"Failed download after retries: {url}") from last_err


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default
