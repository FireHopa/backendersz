from __future__ import annotations

import os
import re
import zipfile
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional


IMAGE_ENGINE_TECHNICAL_SUMMARY_PATH_ENV = "IMAGE_ENGINE_TECHNICAL_SUMMARY_PATH"

# Troque só o nome abaixo se quiser procurar outro arquivo automaticamente em qualquer pasta do backend.
IMAGE_ENGINE_TECHNICAL_SUMMARY_FILENAME_PLACEHOLDER = "resumo_operacional_design_ia_imagem.docx"


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _app_root() -> Path:
    return Path(__file__).resolve().parent


def _normalize_text(text: str, max_len: int = 2200) -> str:
    normalized = re.sub(r"\r\n?", "\n", text or "")
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = normalized.strip()
    return normalized[:max_len]


def _resolve_candidate_paths() -> List[Path]:
    candidates: List[Path] = []
    env_path = (os.getenv(IMAGE_ENGINE_TECHNICAL_SUMMARY_PATH_ENV) or "").strip()
    if env_path:
        raw = Path(env_path)
        candidates.append(raw if raw.is_absolute() else (_backend_root() / raw))

    placeholder_name = (IMAGE_ENGINE_TECHNICAL_SUMMARY_FILENAME_PLACEHOLDER or "").strip()
    if placeholder_name:
        candidates.append(_backend_root() / placeholder_name)
        candidates.append(_app_root() / placeholder_name)

    return candidates


def _read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_docx_file(path: Path) -> str:
    namespaces = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    }

    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    paragraphs: List[str] = []

    for paragraph in root.findall(".//w:p", namespaces):
        parts: List[str] = []
        for node in paragraph.iter():
            tag = node.tag.split("}")[-1]
            if tag == "t" and node.text:
                parts.append(node.text)
            elif tag in {"br", "cr", "tab"}:
                parts.append("\n")
        paragraph_text = "".join(parts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)

    return "\n".join(paragraphs)


def _read_summary_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _read_docx_file(path)
    if suffix in {".txt", ".md", ".markdown", ".csv"}:
        return _read_text_file(path)
    raise ValueError(
        f"Resumo técnico em formato não suportado: {path.name}. Use DOCX, TXT ou MD."
    )


def _find_summary_path() -> Optional[Path]:
    checked = set()
    for candidate in _resolve_candidate_paths():
        resolved = candidate.resolve()
        if str(resolved) in checked:
            continue
        checked.add(str(resolved))
        if resolved.is_file():
            return resolved

    placeholder_name = (IMAGE_ENGINE_TECHNICAL_SUMMARY_FILENAME_PLACEHOLDER or "").strip()
    if not placeholder_name:
        return None

    backend_root = _backend_root().resolve()
    for found in backend_root.rglob(placeholder_name):
        if found.is_file():
            return found.resolve()

    return None


def _extract_section(lines: List[str], heading: str, next_heading: Optional[str] = None) -> List[str]:
    try:
        start_idx = lines.index(heading) + 1
    except ValueError:
        return []

    end_idx = len(lines)
    if next_heading:
        try:
            end_idx = lines.index(next_heading, start_idx)
        except ValueError:
            end_idx = len(lines)

    section = lines[start_idx:end_idx]
    return [item.strip() for item in section if item.strip()]


def _build_compact_summary_prompt(raw_text: str) -> str:
    lines = [line.strip() for line in _normalize_text(raw_text, max_len=12000).split("\n") if line.strip()]

    production_section = _extract_section(
        lines,
        "4. Versão condensada para usar como contexto de produção",
        "5. Checklist mínimo antes de enviar para a engine final",
    )
    priorities_section = _extract_section(
        lines,
        "3. Prioridades que mais valem token no prompt",
        "4. Versão condensada para usar como contexto de produção",
    )
    checklist_section = _extract_section(lines, "5. Checklist mínimo antes de enviar para a engine final")

    production_context = production_section[0] if production_section else ""
    priorities = priorities_section[:8]
    checklist = checklist_section[:8]

    blocks: List[str] = [
        "Aplique obrigatoriamente o resumo técnico estrutural abaixo nesta geração/edição.",
    ]

    if production_context:
        blocks.append(f"Contexto de produção: {production_context}")

    if priorities:
        blocks.append("Prioridades estruturais obrigatórias:")
        blocks.extend(f"- {item}" for item in priorities)

    if checklist:
        blocks.append("Checklist obrigatório antes de considerar a composição correta:")
        blocks.extend(f"- {item.lstrip('☐ ').strip()}" for item in checklist)

    if len(blocks) == 1:
        blocks.append(_normalize_text(raw_text, max_len=1600))

    blocks.append(
        "Essas regras devem orientar foco, hierarquia, grid, agrupamento, contraste, respiro, zonas seguras de texto e balanceamento visual da peça final."
    )

    return _normalize_text("\n".join(blocks), max_len=1900)


@lru_cache(maxsize=8)
def _load_summary_cached(path_str: str, mtime_ns: int, file_size: int) -> str:
    del mtime_ns, file_size
    raw_text = _read_summary_file(Path(path_str))
    return _build_compact_summary_prompt(raw_text)


def load_image_engine_technical_summary() -> Dict[str, str]:
    path = _find_summary_path()
    if not path:
        return {
            "content": "",
            "source_path": "",
            "status": "missing",
        }

    stat = path.stat()
    content = _load_summary_cached(str(path), stat.st_mtime_ns, stat.st_size)
    return {
        "content": content,
        "source_path": str(path),
        "status": "loaded" if content else "empty",
    }


def build_image_engine_technical_summary_block() -> str:
    summary = load_image_engine_technical_summary()
    content = (summary.get("content") or "").strip()
    if not content:
        return ""

    return (
        "Resumo técnico obrigatório do sistema para esta geração/edição:\n"
        f"{content}\n\n"
        "Use este resumo como regra estrutural mandatória e preserve esses princípios na saída final."
    )
