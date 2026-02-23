from __future__ import annotations

import json
from typing import Any, Dict, Optional

from openai import OpenAI
from .config import OPENAI_API_KEY, OPENAI_MODEL


def get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set. Create backend/.env")
    return OpenAI(api_key=OPENAI_API_KEY)


def _strip_fenced(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip().strip("`")
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return t.strip()


def chat_json(*, system: str, user: str, model: Optional[str] = None, temperature: float = 0.3, max_output_tokens: int = 1200) -> Dict[str, Any]:
    """Executa uma chamada e força saída JSON (retorna dict)."""
    client = get_client()
    resp = client.responses.create(
        model=model or OPENAI_MODEL,
        instructions=system + "\n\nREGRAS: responda SOMENTE JSON válido. Sem markdown. Sem texto extra.",
        input=[{"role": "user", "content": user}],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    text = _strip_fenced(resp.output_text or "")
    try:
        data = json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Resposta não é JSON válido: {e}. Texto: {text[:200]}")
    if not isinstance(data, dict):
        raise RuntimeError("JSON retornado não é um objeto (dict).")
    return data
