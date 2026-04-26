from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import requests


class LLMError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    provider: str  # gemini | openai
    model: str
    max_output_tokens: int = 1200
    temperature: float = 0.2


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def load_llm_config() -> LLMConfig:
    provider = _env("LLM_PROVIDER") or ("gemini" if _env("GEMINI_API_KEY") else "openai")
    if provider not in ("gemini", "openai"):
        raise LLMError("LLM_PROVIDER must be 'gemini' or 'openai'")

    if provider == "gemini":
        model = _env("MODEL") or "gemini-1.5-pro"
    else:
        model = _env("MODEL") or "gpt-4.1-mini"

    max_tokens = int(_env("MAX_OUTPUT_TOKENS") or "1200")
    return LLMConfig(provider=provider, model=model, max_output_tokens=max_tokens)


def generate_text(prompt: str, *, cfg: Optional[LLMConfig] = None) -> str:
    cfg = cfg or load_llm_config()
    if cfg.provider == "gemini":
        return _gemini_generate(prompt, cfg=cfg)
    return _openai_generate(prompt, cfg=cfg)


def _gemini_generate(prompt: str, *, cfg: LLMConfig) -> str:
    api_key = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
    if not api_key:
        raise LLMError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY).")

    # Prefer google-generativeai if available (already in requirements-dev.txt).
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:
        raise LLMError(
            "Gemini summarization requires google-generativeai. "
            "Install with: pip install google-generativeai"
        ) from e

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(cfg.model)
    resp = model.generate_content(
        prompt,
        generation_config={
            "temperature": cfg.temperature,
            "max_output_tokens": cfg.max_output_tokens,
        },
    )
    text = getattr(resp, "text", None) or ""
    return str(text).strip()


def _openai_generate(prompt: str, *, cfg: LLMConfig) -> str:
    api_key = _env("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("Missing OPENAI_API_KEY.")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_output_tokens,
        "messages": [
            {"role": "system", "content": "You are a helpful research assistant. Follow the user's formatting constraints strictly."},
            {"role": "user", "content": prompt},
        ],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    try:
        r.raise_for_status()
    except Exception as e:
        raise LLMError(f"OpenAI API error: HTTP {r.status_code}: {r.text[:500]}") from e

    data: Any = r.json()
    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except Exception as e:
        raise LLMError(f"Unexpected OpenAI response: {json.dumps(data)[:500]}") from e

