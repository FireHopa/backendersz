from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlmodel import Session

from .config import YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REDIRECT_URI
from .db import get_session
from .deps import get_current_user
from .models import User

router = APIRouter(prefix="/api/youtube", tags=["youtube"])

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
YOUTUBE_UPLOAD_BASE = "https://www.googleapis.com/upload/youtube/v3"
YOUTUBE_CONNECT_SCOPES = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.upload",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_configured() -> None:
    if not YOUTUBE_CLIENT_ID or not YOUTUBE_CLIENT_SECRET or not YOUTUBE_REDIRECT_URI:
        raise HTTPException(status_code=500, detail="Configuração do YouTube OAuth incompleta no backend.")


def _token_is_expired(expires_at: Optional[datetime]) -> bool:
    if not expires_at:
        return True
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at <= (_utcnow() + timedelta(minutes=2))


def _normalized_scope_set(scope_value: Optional[str]) -> set[str]:
    if not scope_value:
        return set()
    return {item.strip() for item in str(scope_value).split() if item.strip()}


async def _refresh_access_token_if_needed(current_user: User, session: Session) -> str:
    _ensure_configured()
    if current_user.youtube_access_token and not _token_is_expired(current_user.youtube_token_expires_at):
        return current_user.youtube_access_token

    if not current_user.youtube_refresh_token:
        raise HTTPException(status_code=400, detail="YouTube não está conectado nesta conta ou requer reconexão.")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": YOUTUBE_CLIENT_ID,
                "client_secret": YOUTUBE_CLIENT_SECRET,
                "refresh_token": current_user.youtube_refresh_token,
                "grant_type": "refresh_token",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Erro ao renovar token do YouTube: {resp.text}")

    data = resp.json()
    access_token = data.get("access_token")
    expires_in = int(data.get("expires_in", 3600))
    if not access_token:
        raise HTTPException(status_code=400, detail="A Google não retornou um access_token válido para o YouTube.")

    current_user.youtube_access_token = access_token
    current_user.youtube_token_expires_at = _utcnow() + timedelta(seconds=expires_in)
    session.add(current_user)
    session.commit()
    session.refresh(current_user)
    return access_token


async def _fetch_channel(access_token: str) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(
            f"{YOUTUBE_API_BASE}/channels",
            params={"part": "snippet", "mine": "true"},
            headers={"Authorization": f"Bearer {access_token}"},
        )

    if resp.status_code != 200:
        detail_text = resp.text
        try:
            payload = resp.json()
        except Exception:
            payload = None

        reason = None
        message = None
        if isinstance(payload, dict):
            error = payload.get("error") or {}
            message = error.get("message")
            errors = error.get("errors") or []
            if errors and isinstance(errors[0], dict):
                reason = errors[0].get("reason")

        if resp.status_code == 403 and reason in {"insufficientPermissions", "forbidden"}:
            raise HTTPException(
                status_code=400,
                detail=(
                    "A conexão do YouTube foi autorizada sem permissão suficiente para ler o canal. "
                    "Desconecte e conecte novamente para autorizar o escopo completo do YouTube."
                ),
            )

        raise HTTPException(status_code=400, detail=f"Erro ao obter canal do YouTube: {detail_text}")

    data = resp.json()
    items = data.get("items") or []
    if not items:
        raise HTTPException(status_code=400, detail="A conta Google conectada não possui um canal do YouTube disponível.")
    item = items[0]
    snippet = item.get("snippet") or {}
    custom_url = snippet.get("customUrl")
    handle = None
    if custom_url:
        handle = custom_url if str(custom_url).startswith("@") else f"@{custom_url}"
    return {
        "channel_id": item.get("id"),
        "channel_title": snippet.get("title"),
        "channel_handle": handle,
        "channel_thumbnail": (((snippet.get("thumbnails") or {}).get("default") or {}).get("url")),
    }


@router.get("/auth-url")
def get_auth_url(state: Optional[str] = None, current_user: User = Depends(get_current_user)):
    _ensure_configured()
    final_state = state or secrets.token_urlsafe(24)
    params = {
        "client_id": YOUTUBE_CLIENT_ID,
        "redirect_uri": YOUTUBE_REDIRECT_URI,
        "response_type": "code",
        "access_type": "offline",
        "include_granted_scopes": "true",
        "prompt": "consent",
        "scope": " ".join(YOUTUBE_CONNECT_SCOPES),
        "state": final_state,
    }
    qs = httpx.QueryParams(params)
    return {"url": f"https://accounts.google.com/o/oauth2/v2/auth?{qs}"}


@router.post("/connect")
async def connect_youtube(payload: dict, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    _ensure_configured()
    code = str(payload.get("code") or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="Código de autorização do YouTube não informado.")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": YOUTUBE_CLIENT_ID,
                "client_secret": YOUTUBE_CLIENT_SECRET,
                "redirect_uri": YOUTUBE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Erro ao conectar YouTube: {resp.text}")

    token_data = resp.json()
    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token") or current_user.youtube_refresh_token
    expires_in = int(token_data.get("expires_in", 3600))
    granted_scopes = _normalized_scope_set(token_data.get("scope"))
    if not access_token:
        raise HTTPException(status_code=400, detail="A Google não retornou um access_token válido.")

    if "https://www.googleapis.com/auth/youtube" not in granted_scopes:
        raise HTTPException(
            status_code=400,
            detail=(
                "A autorização do YouTube voltou sem o escopo completo de canal. "
                "Desconecte a conta, remova o acesso anterior do app no Google e conecte novamente."
            ),
        )

    channel = await _fetch_channel(access_token)

    current_user.youtube_access_token = access_token
    current_user.youtube_refresh_token = refresh_token
    current_user.youtube_token_expires_at = _utcnow() + timedelta(seconds=expires_in)
    current_user.youtube_channel_id = channel["channel_id"]
    current_user.youtube_channel_title = channel["channel_title"]
    current_user.youtube_channel_handle = channel["channel_handle"]
    current_user.youtube_channel_thumbnail = channel["channel_thumbnail"]
    session.add(current_user)
    session.commit()

    return {
        "ok": True,
        "message": "Canal do YouTube conectado com sucesso!",
        **channel,
    }


@router.get("/status")
async def youtube_status(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    connected = bool(current_user.youtube_channel_id and current_user.youtube_refresh_token)
    if connected:
        try:
            access_token = await _refresh_access_token_if_needed(current_user, session)
            channel = await _fetch_channel(access_token)
            current_user.youtube_channel_id = channel["channel_id"]
            current_user.youtube_channel_title = channel["channel_title"]
            current_user.youtube_channel_handle = channel["channel_handle"]
            current_user.youtube_channel_thumbnail = channel["channel_thumbnail"]
            session.add(current_user)
            session.commit()
            session.refresh(current_user)
        except HTTPException:
            pass

    return {
        "connected": connected,
        "channel_id": current_user.youtube_channel_id,
        "channel_title": current_user.youtube_channel_title,
        "channel_handle": current_user.youtube_channel_handle,
        "channel_thumbnail": current_user.youtube_channel_thumbnail,
    }


@router.post("/disconnect")
def disconnect_youtube(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    current_user.youtube_access_token = None
    current_user.youtube_refresh_token = None
    current_user.youtube_token_expires_at = None
    current_user.youtube_channel_id = None
    current_user.youtube_channel_title = None
    current_user.youtube_channel_handle = None
    current_user.youtube_channel_thumbnail = None
    session.add(current_user)
    session.commit()
    return {"ok": True}


@router.post("/publish")
async def publish_youtube(
    title: str = Form(...),
    description: str = Form(""),
    privacy_status: str = Form("private"),
    made_for_kids: bool = Form(False),
    tags: str = Form(""),
    category_id: Optional[str] = Form("22"),
    video_file: UploadFile = File(...),
    thumbnail_file: Optional[UploadFile] = File(None),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    access_token = await _refresh_access_token_if_needed(current_user, session)

    valid_privacy = {"private", "public", "unlisted"}
    if privacy_status not in valid_privacy:
        raise HTTPException(status_code=400, detail="privacy_status inválido. Use private, public ou unlisted.")

    clean_title = title.strip()
    if not clean_title:
        raise HTTPException(status_code=400, detail="O título do vídeo é obrigatório.")

    if not video_file.filename:
        raise HTTPException(status_code=400, detail="Nenhum arquivo de vídeo foi enviado.")

    video_stream = video_file.file
    video_stream.seek(0, os.SEEK_END)
    video_size = video_stream.tell()
    video_stream.seek(0)
    if not video_size:
        raise HTTPException(status_code=400, detail="O arquivo de vídeo está vazio.")

    snippet: dict = {
        "title": clean_title,
        "description": description or "",
        "categoryId": str(category_id or "22"),
    }
    parsed_tags = [item.strip() for item in (tags or "").split(",") if item.strip()]
    if parsed_tags:
        snippet["tags"] = parsed_tags[:500]

    metadata = {
        "snippet": snippet,
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": bool(made_for_kids),
        },
    }

    init_headers = {
        "Authorization": f"Bearer {access_token}",
        "X-Upload-Content-Length": str(video_size),
        "X-Upload-Content-Type": video_file.content_type or "application/octet-stream",
        "Content-Type": "application/json; charset=UTF-8",
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        init_resp = await client.post(
            f"{YOUTUBE_UPLOAD_BASE}/videos",
            params={"uploadType": "resumable", "part": "snippet,status"},
            headers=init_headers,
            content=json.dumps(metadata),
        )

    if init_resp.status_code not in {200, 201}:
        raise HTTPException(status_code=400, detail=f"Erro ao iniciar upload do vídeo no YouTube: {init_resp.text}")

    upload_url = init_resp.headers.get("Location") or init_resp.headers.get("location")
    if not upload_url:
        raise HTTPException(status_code=400, detail="O YouTube não retornou a URL de upload resumable.")

    video_bytes = await video_file.read()
    upload_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": video_file.content_type or "application/octet-stream",
        "Content-Length": str(len(video_bytes)),
    }

    async with httpx.AsyncClient(timeout=3600.0) as client:
        upload_resp = await client.put(upload_url, headers=upload_headers, content=video_bytes)

    if upload_resp.status_code not in {200, 201}:
        raise HTTPException(status_code=400, detail=f"Erro ao enviar vídeo para o YouTube: {upload_resp.text}")

    upload_data = upload_resp.json()
    video_id = upload_data.get("id")
    if not video_id:
        raise HTTPException(status_code=400, detail="O YouTube não retornou o ID do vídeo publicado.")

    thumbnail_warning = None
    if thumbnail_file and thumbnail_file.filename:
        thumb_bytes = await thumbnail_file.read()
        if thumb_bytes:
            thumb_headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": thumbnail_file.content_type or "application/octet-stream",
                "Content-Length": str(len(thumb_bytes)),
            }
            async with httpx.AsyncClient(timeout=300.0) as client:
                thumb_resp = await client.post(
                    f"{YOUTUBE_UPLOAD_BASE}/thumbnails/set",
                    params={"videoId": video_id, "uploadType": "media"},
                    headers=thumb_headers,
                    content=thumb_bytes,
                )
            if thumb_resp.status_code not in {200, 201}:
                thumbnail_warning = f"Vídeo publicado, mas houve falha ao enviar a thumbnail: {thumb_resp.text}"

    video_url = f"https://www.youtube.com/watch?v={video_id}"
    return {
        "ok": True,
        "message": "Vídeo publicado no YouTube com sucesso!",
        "video_id": video_id,
        "video_url": video_url,
        "channel_title": current_user.youtube_channel_title,
        "thumbnail_warning": thumbnail_warning,
    }
