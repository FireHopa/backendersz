from __future__ import annotations

import json
import re
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from .db import get_session
from .deps import get_current_user
from .models import BobarCard, BobarColumn, User, utcnow

router = APIRouter(prefix="/api/bobar", tags=["Bobar"])

DEFAULT_COLUMNS = ("Entrada", "Em produção", "Finalizados")
CARD_TYPES = {"manual", "roteiro", "conteudo", "ideia", "checklist", "fluxograma"}

TIME_RANGE_RE = re.compile(r"(?i)\b(\d+\s*(?:-|a|até|to)\s*\d+\s*s)\b")
TIME_SINGLE_RE = re.compile(r"(?i)\b(\d+\s*s)\b")


class BobarCardOut(BaseModel):
    id: int
    column_id: int
    title: str
    card_type: str
    source_kind: Optional[str] = None
    source_label: Optional[str] = None
    content_text: str
    note: str
    position: int
    structure_json: str
    created_at: str
    updated_at: str


class BobarColumnOut(BaseModel):
    id: int
    name: str
    position: int
    cards: list[BobarCardOut]


class BobarBoardOut(BaseModel):
    title: str
    total_cards: int
    columns: list[BobarColumnOut]


class BobarColumnCreateIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


class BobarColumnUpdateIn(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)


class BobarCardCreateIn(BaseModel):
    column_id: Optional[int] = None
    title: Optional[str] = Field(default=None, max_length=140)
    note: str = Field(default="", max_length=5000)
    content_text: str = Field(default="", max_length=200000)
    card_type: Optional[str] = Field(default=None, max_length=40)
    source_kind: Optional[str] = Field(default=None, max_length=60)
    source_label: Optional[str] = Field(default=None, max_length=120)
    structure_json: Optional[str] = Field(default=None, max_length=500000)


class BobarCardUpdateIn(BaseModel):
    title: Optional[str] = Field(default=None, max_length=140)
    note: Optional[str] = Field(default=None, max_length=5000)
    content_text: Optional[str] = Field(default=None, max_length=200000)
    card_type: Optional[str] = Field(default=None, max_length=40)
    column_id: Optional[int] = None
    structure_json: Optional[str] = Field(default=None, max_length=500000)


class BobarCardMoveIn(BaseModel):
    column_id: int
    position: int = Field(default=0, ge=0)


def _clean_text(value: Optional[str]) -> str:
    return str(value or "").strip()


def _clip(value: str, limit: int) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def _parse_json(value: str):
    text = _clean_text(value)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_first_meaningful_line(text: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^[#>*\-\d.\)\s]+", "", line).strip()
        if line:
            return line
    return ""


def _derive_card_type(explicit_card_type: Optional[str], content_text: str) -> str:
    normalized = _clean_text(explicit_card_type).lower().replace(" ", "_")
    if normalized in CARD_TYPES:
        return normalized

    parsed = _parse_json(content_text)
    if isinstance(parsed, dict):
        if any(key in parsed for key in ("roteiro_segundo_a_segundo", "hooks", "legenda", "estrategia_do_video")):
            return "roteiro"
        if "blocos" in parsed:
            return "conteudo"
        if any(key in parsed for key in ("checklist", "passos", "steps")):
            return "checklist"

    lower = content_text.lower()
    if "hook" in lower or "roteiro" in lower or "legenda" in lower:
        return "roteiro"
    if "checklist" in lower:
        return "checklist"
    if content_text.strip():
        return "conteudo"
    return "manual"


def _derive_card_title(title: Optional[str], source_label: Optional[str], content_text: str) -> str:
    explicit = _clean_text(title)
    if explicit:
        return _clip(explicit, 140)

    parsed = _parse_json(content_text)
    if isinstance(parsed, dict):
        for key in (
            "titulo_da_tela",
            "title",
            "titulo",
            "headline",
            "tema",
            "assunto",
            "requested_task",
            "selected_theme",
        ):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return _clip(value.strip(), 140)

        hooks = parsed.get("hooks")
        if isinstance(hooks, list) and hooks:
            first_hook = _clean_text(hooks[0])
            if first_hook:
                return _clip(first_hook, 140)

    first_line = _extract_first_meaningful_line(content_text)
    if first_line:
        return _clip(first_line, 140)

    if _clean_text(source_label):
        return _clip(_clean_text(source_label), 140)

    return "Novo card"


def _json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _new_node_id() -> str:
    return f"node-{uuid.uuid4().hex[:10]}"


def _new_edge_id() -> str:
    return f"edge-{uuid.uuid4().hex[:10]}"


def _build_node(title: str, content: str, index: int, time_label: str = "", kind: str = "step") -> dict[str, Any]:
    safe_title = _clip(_clean_text(title) or f"Bloco {index + 1}", 80)
    safe_content = content.strip()
    return {
        "id": _new_node_id(),
        "title": safe_title,
        "content": safe_content,
        "time": _clip(_clean_text(time_label), 40),
        "kind": _clean_text(kind).lower() or "step",
        "x": 80 + (index * 280),
        "y": 90 + ((index % 2) * 180),
    }


def _normalize_node(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return _build_node(f"Bloco {index + 1}", "", index)

    title = _clip(_clean_text(raw.get("title") or raw.get("label") or raw.get("name") or f"Bloco {index + 1}"), 80)
    content = str(raw.get("content") or raw.get("description") or raw.get("text") or "").strip()
    time_label = _clip(_clean_text(raw.get("time") or raw.get("tempo")), 40)
    kind = _clean_text(raw.get("kind") or raw.get("type") or "step").lower() or "step"

    try:
        x = int(raw.get("x", 80 + (index * 280)))
    except Exception:
        x = 80 + (index * 280)

    try:
        y = int(raw.get("y", 90 + ((index % 2) * 180)))
    except Exception:
        y = 90 + ((index % 2) * 180)

    return {
        "id": _clean_text(raw.get("id")) or _new_node_id(),
        "title": title,
        "content": content,
        "time": time_label,
        "kind": kind,
        "x": max(0, x),
        "y": max(0, y),
    }


def _normalize_edge(raw: Any, valid_ids: set[str]) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None

    source = _clean_text(raw.get("source"))
    target = _clean_text(raw.get("target"))
    if not source or not target:
        return None
    if source not in valid_ids or target not in valid_ids or source == target:
        return None

    return {
        "id": _clean_text(raw.get("id")) or _new_edge_id(),
        "source": source,
        "target": target,
        "label": _clip(_clean_text(raw.get("label")), 60),
    }


def _edges_from_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for index in range(len(nodes) - 1):
        edges.append(
            {
                "id": _new_edge_id(),
                "source": nodes[index]["id"],
                "target": nodes[index + 1]["id"],
                "label": "",
            }
        )
    return edges


def _build_nodes_from_script_json(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    index = 0

    hooks = parsed.get("hooks")
    if isinstance(hooks, list):
        for hook in hooks:
            hook_text = _clean_text(hook if isinstance(hook, str) else json.dumps(hook, ensure_ascii=False))
            if not hook_text:
                continue
            nodes.append(_build_node(f"Hook {len(nodes) + 1}", hook_text, index, kind="hook"))
            index += 1

    second_by_second = parsed.get("roteiro_segundo_a_segundo")
    if isinstance(second_by_second, list):
        for item in second_by_second:
            if isinstance(item, dict):
                time_label = _clean_text(item.get("tempo"))
                action = _clean_text(item.get("acao"))
                speech = _clean_text(item.get("fala"))
                title = time_label or f"Bloco {index + 1}"
                parts = []
                if action:
                    parts.append(f"Ação: {action}")
                if speech:
                    parts.append(f"Fala: {speech}")
                content = "\n\n".join(parts).strip() or json.dumps(item, ensure_ascii=False)
            else:
                time_label = ""
                title = f"Bloco {index + 1}"
                content = _clean_text(item if isinstance(item, str) else json.dumps(item, ensure_ascii=False))

            if content:
                nodes.append(_build_node(title, content, index, time_label=time_label, kind="timeline"))
                index += 1

    blocks = parsed.get("blocos")
    if isinstance(blocks, list) and not nodes:
        for block in blocks:
            if isinstance(block, dict):
                title = _clean_text(block.get("titulo") or block.get("title") or f"Bloco {index + 1}")
                content = _clean_text(block.get("conteudo") or block.get("content") or "")
            else:
                title = f"Bloco {index + 1}"
                content = _clean_text(str(block))
            if content or title:
                nodes.append(_build_node(title, content, index, kind="step"))
                index += 1

    screen_texts = parsed.get("texto_na_tela")
    if isinstance(screen_texts, list) and screen_texts:
        joined = "\n".join(_clean_text(item) for item in screen_texts if _clean_text(str(item)))
        if joined:
            nodes.append(_build_node("Texto na tela", joined, index, kind="support"))
            index += 1

    legend = _clean_text(parsed.get("legenda") if isinstance(parsed.get("legenda"), str) else "")
    if legend:
        nodes.append(_build_node("Legenda", legend, index, kind="cta"))

    return nodes


def _build_nodes_from_text(content_text: str) -> list[dict[str, Any]]:
    text = content_text.strip()
    if not text:
        return []

    lines = [line.rstrip() for line in text.splitlines()]
    segments: list[tuple[str, str]] = []
    buffer: list[str] = []
    current_time = ""

    def flush():
        nonlocal buffer, current_time
        body = "\n".join(part for part in buffer if part.strip()).strip()
        if body:
            segments.append((current_time, body))
        buffer = []
        current_time = ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if buffer:
                flush()
            continue

        time_match = TIME_RANGE_RE.search(line) or TIME_SINGLE_RE.search(line)
        starts_new_segment = bool(
            time_match
            or re.match(r"^(hook|abertura|cta|call to action|encerramento|cena|bloco)\b", line.lower())
            or re.match(r"^\d+[\).\-\:]", line)
        )

        if starts_new_segment and buffer:
            flush()

        if time_match:
            current_time = _clean_text(time_match.group(1)).replace(" ", "")

        buffer.append(line)

    if buffer:
        flush()

    if not segments:
        paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        if not paragraphs:
            paragraphs = [line.strip() for line in lines if line.strip()]
        segments = [("", paragraph) for paragraph in paragraphs]

    nodes: list[dict[str, Any]] = []
    for index, (time_label, body) in enumerate(segments):
        first_line = _extract_first_meaningful_line(body)
        title = time_label or _clip(first_line or f"Bloco {index + 1}", 80)
        kind = "timeline" if time_label else "step"
        nodes.append(_build_node(title, body, index, time_label=time_label, kind=kind))
    return nodes


def _normalize_flowchart_structure(raw_structure: Any, title: str, content_text: str) -> dict[str, Any]:
    parsed = raw_structure
    if isinstance(raw_structure, str):
        parsed = _parse_json(raw_structure) or {}

    nodes: list[dict[str, Any]] = []
    if isinstance(parsed, dict):
        raw_nodes = parsed.get("nodes")
        if isinstance(raw_nodes, list):
            nodes = [_normalize_node(node, index) for index, node in enumerate(raw_nodes)]

    if not nodes:
        content_json = _parse_json(content_text)
        if isinstance(content_json, dict):
            nodes = _build_nodes_from_script_json(content_json)
        if not nodes:
            nodes = _build_nodes_from_text(content_text)

    if not nodes:
        nodes = [_build_node(title or "Início", content_text.strip(), 0, kind="step")]

    valid_ids = {node["id"] for node in nodes}
    edges: list[dict[str, Any]] = []
    if isinstance(parsed, dict):
        raw_edges = parsed.get("edges")
        if isinstance(raw_edges, list):
            edges = [edge for edge in (_normalize_edge(item, valid_ids) for item in raw_edges) if edge]

    if not edges and len(nodes) > 1:
        edges = _edges_from_nodes(nodes)

    return {
        "nodes": nodes,
        "edges": edges,
    }


def _resolve_structure_json(card_type: str, provided_structure_json: Optional[str], title: str, content_text: str) -> str:
    if card_type != "fluxograma" and not _clean_text(provided_structure_json):
        return "{}"

    normalized = _normalize_flowchart_structure(provided_structure_json or "{}", title, content_text)
    return _json_dumps(normalized)


def _ensure_default_columns(session: Session, current_user: User) -> list[BobarColumn]:
    columns = session.exec(
        select(BobarColumn)
        .where(BobarColumn.user_id == current_user.id)
        .order_by(BobarColumn.position.asc(), BobarColumn.id.asc())
    ).all()

    if columns:
        return columns

    now = utcnow()
    for index, name in enumerate(DEFAULT_COLUMNS):
        session.add(
            BobarColumn(
                user_id=current_user.id,
                name=name,
                position=index,
                created_at=now,
                updated_at=now,
            )
        )
    session.commit()

    return session.exec(
        select(BobarColumn)
        .where(BobarColumn.user_id == current_user.id)
        .order_by(BobarColumn.position.asc(), BobarColumn.id.asc())
    ).all()


def _reindex_columns(session: Session, current_user: User) -> None:
    columns = session.exec(
        select(BobarColumn)
        .where(BobarColumn.user_id == current_user.id)
        .order_by(BobarColumn.position.asc(), BobarColumn.id.asc())
    ).all()

    changed = False
    for index, column in enumerate(columns):
        if column.position != index:
            column.position = index
            column.updated_at = utcnow()
            session.add(column)
            changed = True

    if changed:
        session.commit()


def _reindex_cards_for_column(session: Session, current_user: User, column_id: int) -> None:
    cards = session.exec(
        select(BobarCard)
        .where(BobarCard.user_id == current_user.id, BobarCard.column_id == column_id)
        .order_by(BobarCard.position.asc(), BobarCard.id.asc())
    ).all()

    changed = False
    for index, card in enumerate(cards):
        if card.position != index:
            card.position = index
            card.updated_at = utcnow()
            session.add(card)
            changed = True

    if changed:
        session.commit()


def _get_column_or_404(session: Session, current_user: User, column_id: int) -> BobarColumn:
    column = session.get(BobarColumn, column_id)
    if not column or column.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Coluna do Bobar não encontrada.")
    return column


def _get_card_or_404(session: Session, current_user: User, card_id: int) -> BobarCard:
    card = session.get(BobarCard, card_id)
    if not card or card.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Card do Bobar não encontrado.")
    return card


def _card_out(card: BobarCard) -> BobarCardOut:
    structure_json = card.structure_json or "{}"
    if (card.card_type or "").lower() == "fluxograma":
        structure_json = _resolve_structure_json("fluxograma", structure_json, card.title, card.content_text)

    return BobarCardOut(
        id=card.id or 0,
        column_id=card.column_id,
        title=card.title,
        card_type=card.card_type,
        source_kind=card.source_kind,
        source_label=card.source_label,
        content_text=card.content_text,
        note=card.note,
        position=card.position,
        structure_json=structure_json,
        created_at=card.created_at.isoformat(),
        updated_at=card.updated_at.isoformat(),
    )


def _build_board(session: Session, current_user: User) -> BobarBoardOut:
    columns = _ensure_default_columns(session, current_user)
    cards = session.exec(
        select(BobarCard)
        .where(BobarCard.user_id == current_user.id)
        .order_by(BobarCard.column_id.asc(), BobarCard.position.asc(), BobarCard.id.asc())
    ).all()

    by_column: dict[int, list[BobarCardOut]] = {column.id: [] for column in columns}
    for card in cards:
        by_column.setdefault(card.column_id, []).append(_card_out(card))

    return BobarBoardOut(
        title="Bobar",
        total_cards=len(cards),
        columns=[
            BobarColumnOut(
                id=column.id or 0,
                name=column.name,
                position=column.position,
                cards=by_column.get(column.id or 0, []),
            )
            for column in columns
        ],
    )


@router.get("", response_model=BobarBoardOut)
def bobar_board(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    return _build_board(session, current_user)


@router.post("/columns", response_model=BobarBoardOut)
def bobar_create_column(
    payload: BobarColumnCreateIn,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    columns = _ensure_default_columns(session, current_user)
    name = _clean_text(payload.name)
    if not name:
        raise HTTPException(status_code=400, detail="Informe um nome para a coluna.")

    session.add(
        BobarColumn(
            user_id=current_user.id,
            name=_clip(name, 80),
            position=len(columns),
            created_at=utcnow(),
            updated_at=utcnow(),
        )
    )
    session.commit()
    return _build_board(session, current_user)


@router.patch("/columns/{column_id}", response_model=BobarBoardOut)
def bobar_update_column(
    column_id: int,
    payload: BobarColumnUpdateIn,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    column = _get_column_or_404(session, current_user, column_id)

    if payload.name is not None:
        name = _clean_text(payload.name)
        if not name:
            raise HTTPException(status_code=400, detail="O nome da coluna não pode ficar vazio.")
        column.name = _clip(name, 80)

    column.updated_at = utcnow()
    session.add(column)
    session.commit()
    return _build_board(session, current_user)


@router.delete("/columns/{column_id}", response_model=BobarBoardOut)
def bobar_delete_column(
    column_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    columns = _ensure_default_columns(session, current_user)
    target = _get_column_or_404(session, current_user, column_id)

    if len(columns) <= 1:
        raise HTTPException(status_code=400, detail="Você precisa manter ao menos uma coluna no Bobar.")

    destination = next((column for column in columns if column.id != target.id), None)
    if destination is None:
        raise HTTPException(status_code=400, detail="Não foi possível realocar os cards desta coluna.")

    destination_cards = session.exec(
        select(BobarCard)
        .where(BobarCard.user_id == current_user.id, BobarCard.column_id == destination.id)
        .order_by(BobarCard.position.asc(), BobarCard.id.asc())
    ).all()
    next_position = len(destination_cards)

    target_cards = session.exec(
        select(BobarCard)
        .where(BobarCard.user_id == current_user.id, BobarCard.column_id == target.id)
        .order_by(BobarCard.position.asc(), BobarCard.id.asc())
    ).all()

    now = utcnow()
    for card in target_cards:
        card.column_id = destination.id or 0
        card.position = next_position
        card.updated_at = now
        next_position += 1
        session.add(card)

    session.delete(target)
    session.commit()
    _reindex_columns(session, current_user)
    _reindex_cards_for_column(session, current_user, destination.id or 0)
    return _build_board(session, current_user)


@router.post("/cards", response_model=BobarBoardOut)
def bobar_create_card(
    payload: BobarCardCreateIn,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    columns = _ensure_default_columns(session, current_user)
    column = _get_column_or_404(session, current_user, payload.column_id) if payload.column_id else columns[0]

    existing_cards = session.exec(
        select(BobarCard)
        .where(BobarCard.user_id == current_user.id, BobarCard.column_id == column.id)
        .order_by(BobarCard.position.asc(), BobarCard.id.asc())
    ).all()

    content_text = payload.content_text or ""
    note = payload.note or ""
    title = _derive_card_title(payload.title, payload.source_label, content_text or note)
    card_type = _derive_card_type(payload.card_type, content_text)
    structure_json = _resolve_structure_json(card_type, payload.structure_json, title, content_text)

    session.add(
        BobarCard(
            user_id=current_user.id,
            column_id=column.id or 0,
            title=title,
            card_type=card_type,
            source_kind=_clean_text(payload.source_kind) or None,
            source_label=_clean_text(payload.source_label) or None,
            content_text=content_text,
            note=note,
            position=len(existing_cards),
            structure_json=structure_json,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
    )
    session.commit()
    return _build_board(session, current_user)


@router.post("/cards/import", response_model=BobarBoardOut)
def bobar_import_card(
    payload: BobarCardCreateIn,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    return bobar_create_card(payload, session, current_user)


@router.patch("/cards/{card_id}", response_model=BobarBoardOut)
def bobar_update_card(
    card_id: int,
    payload: BobarCardUpdateIn,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    card = _get_card_or_404(session, current_user, card_id)
    old_column_id = card.column_id

    if payload.column_id is not None and payload.column_id != card.column_id:
        new_column = _get_column_or_404(session, current_user, payload.column_id)
        existing_cards = session.exec(
            select(BobarCard)
            .where(BobarCard.user_id == current_user.id, BobarCard.column_id == new_column.id)
            .order_by(BobarCard.position.asc(), BobarCard.id.asc())
        ).all()
        card.column_id = new_column.id or 0
        card.position = len(existing_cards)

    if payload.title is not None:
        title = _clean_text(payload.title)
        if not title:
            raise HTTPException(status_code=400, detail="O título do card não pode ficar vazio.")
        card.title = _clip(title, 140)

    if payload.note is not None:
        card.note = payload.note

    if payload.content_text is not None:
        card.content_text = payload.content_text

    if payload.card_type is not None:
        card.card_type = _derive_card_type(payload.card_type, payload.content_text or card.content_text)

    if payload.structure_json is not None:
        card.structure_json = _resolve_structure_json(
            card.card_type,
            payload.structure_json,
            card.title,
            payload.content_text if payload.content_text is not None else card.content_text,
        )
    elif card.card_type == "fluxograma" and not _clean_text(card.structure_json):
        card.structure_json = _resolve_structure_json(
            "fluxograma",
            card.structure_json,
            card.title,
            payload.content_text if payload.content_text is not None else card.content_text,
        )

    card.updated_at = utcnow()
    session.add(card)
    session.commit()

    if old_column_id != card.column_id:
        _reindex_cards_for_column(session, current_user, old_column_id)
        _reindex_cards_for_column(session, current_user, card.column_id)

    return _build_board(session, current_user)


@router.post("/cards/{card_id}/move", response_model=BobarBoardOut)
def bobar_move_card(
    card_id: int,
    payload: BobarCardMoveIn,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    card = _get_card_or_404(session, current_user, card_id)
    target_column = _get_column_or_404(session, current_user, payload.column_id)
    source_column_id = card.column_id
    destination_column_id = target_column.id or 0
    now = utcnow()

    source_cards = session.exec(
        select(BobarCard)
        .where(BobarCard.user_id == current_user.id, BobarCard.column_id == source_column_id)
        .order_by(BobarCard.position.asc(), BobarCard.id.asc())
    ).all()

    if destination_column_id == source_column_id:
        ordered = [item for item in source_cards if item.id != card.id]
        target_position = min(max(payload.position, 0), len(ordered))
        ordered.insert(target_position, card)
        for index, item in enumerate(ordered):
            item.column_id = destination_column_id
            item.position = index
            item.updated_at = now
            session.add(item)
    else:
        source_remaining = [item for item in source_cards if item.id != card.id]
        destination_cards = session.exec(
            select(BobarCard)
            .where(BobarCard.user_id == current_user.id, BobarCard.column_id == destination_column_id)
            .order_by(BobarCard.position.asc(), BobarCard.id.asc())
        ).all()

        target_position = min(max(payload.position, 0), len(destination_cards))
        destination_cards.insert(target_position, card)

        for index, item in enumerate(source_remaining):
            item.position = index
            item.updated_at = now
            session.add(item)

        for index, item in enumerate(destination_cards):
            item.column_id = destination_column_id
            item.position = index
            item.updated_at = now
            session.add(item)

    session.commit()
    return _build_board(session, current_user)


@router.post("/cards/{card_id}/transform-to-flowchart", response_model=BobarBoardOut)
def bobar_transform_to_flowchart(
    card_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    card = _get_card_or_404(session, current_user, card_id)
    card.card_type = "fluxograma"
    card.structure_json = _resolve_structure_json("fluxograma", card.structure_json, card.title, card.content_text)
    card.updated_at = utcnow()
    session.add(card)
    session.commit()
    return _build_board(session, current_user)


@router.delete("/cards/{card_id}", response_model=BobarBoardOut)
def bobar_delete_card(
    card_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    card = _get_card_or_404(session, current_user, card_id)
    column_id = card.column_id
    session.delete(card)
    session.commit()
    _reindex_cards_for_column(session, current_user, column_id)
    return _build_board(session, current_user)
