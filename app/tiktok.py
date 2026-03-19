from __future__ import annotations

import math
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlmodel import Session

from .config import TIKTOK_CLIENT_KEY, TIKTOK_CLIENT_SECRET, TIKTOK_REDIRECT_URI
from .db import get_session
from .deps import get_current_user
from .models import User

router = APIRouter(prefix="/api/tiktok", tags=["tiktok"])

TIKTOK_AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/"
TIKTOK_TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
TIKTOK_REVOKE_URL = "https://open.tiktokapis.com/v2/oauth/revoke/"
TIKTOK_USER_INFO_URL = "https://open.tiktokapis.com/v2/user/info/"
TIKTOK_CREATOR_INFO_URL = "https://open.tiktokapis.com/v2/post/publish/creator_info/query/"
TIKTOK_VIDEO_INIT_URL = "https://open.tiktokapis.com/v2/post/publish/video/init/"
TIKTOK_STATUS_URL = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"

TIKTOK_CONNECT_SCOPES = [
    "user.info.basic",
    "user.info.profile",
    "video.publish",
]

DEFAULT_PRIVACY_OPTIONS = [
    "PUBLIC_TO_EVERYONE",
    "FOLLOWER_OF_CREATOR",
    "MUTUAL_FOLLOW_FRIENDS",
    "SELF_ONLY",
]

PRIVACY_LABELS = {
    "PUBLIC_TO_EVERYONE": "Público",
    "FOLLOWER_OF_CREATOR": "Seguidores",
    "MUTUAL_FOLLOW_FRIENDS": "Amigos mútuos",
    "SELF_ONLY": "Somente eu",
}


class TikTokApiError(Exception):
    def __init__(self, message: str, *, status_code: int = 400, payload: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload or {}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_configured() -> None:
    if not TIKTOK_CLIENT_KEY or not TIKTOK_CLIENT_SECRET or not TIKTOK_REDIRECT_URI:
        raise HTTPException(status_code=500, detail="Configuração do TikTok OAuth incompleta no backend.")


def _token_is_expired(expires_at: Optional[datetime]) -> bool:
    if not expires_at:
        return True
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at <= (_utcnow() + timedelta(minutes=5))


def _normalize_scope_value(value: Optional[str]) -> str:
    if not value:
        return ""
    parts = [part.strip() for part in str(value).replace(" ", ",").split(",") if part.strip()]
    deduped: list[str] = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return ",".join(deduped)


def _scope_set(value: Optional[str]) -> set[str]:
    return {part.strip() for part in _normalize_scope_value(value).split(",") if part.strip()}


def _raise_from_tiktok_response(resp: httpx.Response, fallback_message: str) -> None:
    try:
        payload = resp.json()
    except Exception:
        payload = None

    message = fallback_message
    if isinstance(payload, dict):
        error = payload.get("error") or {}
        detail = error.get("message") or payload.get("message")
        code = error.get("code")
        if detail:
            message = f"{fallback_message}: {detail}"
        if code == "scope_not_authorized":
            message = (
                "O TikTok recusou a operação porque o app ou a conta não têm o escopo necessário. "
                "Confirme se o app foi aprovado para user.info.profile e video.publish e reconecte a conta."
            )
        elif code == "access_token_invalid":
            message = "O token do TikTok expirou ou ficou inválido. Reconecte a conta do TikTok."
        elif code == "privacy_level_option_mismatch":
            message = "A privacidade escolhida não é permitida para essa conta do TikTok no momento."
        raise TikTokApiError(message, status_code=400, payload=payload)

    raise TikTokApiError(f"{fallback_message}: {resp.text}", status_code=400)


async def _post_form(url: str, data: dict) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if resp.status_code != 200:
        _raise_from_tiktok_response(resp, "Erro na autenticação do TikTok")
    return resp.json()


async def _refresh_access_token_if_needed(current_user: User, session: Session) -> str:
    _ensure_configured()
    if current_user.tiktok_access_token and not _token_is_expired(current_user.tiktok_token_expires_at):
        return current_user.tiktok_access_token

    if not current_user.tiktok_refresh_token:
        raise HTTPException(status_code=400, detail="TikTok não está conectado nesta conta ou requer reconexão.")

    data = await _post_form(
        TIKTOK_TOKEN_URL,
        {
            "client_key": TIKTOK_CLIENT_KEY,
            "client_secret": TIKTOK_CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": current_user.tiktok_refresh_token,
        },
    )

    access_token = data.get("access_token")
    refresh_token = data.get("refresh_token") or current_user.tiktok_refresh_token
    expires_in = int(data.get("expires_in", 86400))
    refresh_expires_in = int(data.get("refresh_expires_in", 31536000))
    open_id = data.get("open_id") or current_user.tiktok_open_id
    granted_scope = _normalize_scope_value(data.get("scope") or current_user.tiktok_scope)

    if not access_token:
        raise HTTPException(status_code=400, detail="O TikTok não retornou um access_token válido.")

    current_user.tiktok_access_token = access_token
    current_user.tiktok_refresh_token = refresh_token
    current_user.tiktok_token_expires_at = _utcnow() + timedelta(seconds=expires_in)
    current_user.tiktok_refresh_token_expires_at = _utcnow() + timedelta(seconds=refresh_expires_in)
    current_user.tiktok_open_id = open_id
    current_user.tiktok_scope = granted_scope
    session.add(current_user)
    session.commit()
    session.refresh(current_user)
    return access_token


async def _fetch_user_profile(access_token: str) -> dict:
    params = {
        "fields": "open_id,display_name,avatar_url,profile_deep_link,username,bio_description,is_verified",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(
            TIKTOK_USER_INFO_URL,
            params=params,
            headers={"Authorization": f"Bearer {access_token}"},
        )
    if resp.status_code != 200:
        _raise_from_tiktok_response(resp, "Erro ao obter perfil do TikTok")

    payload = resp.json()
    data = payload.get("data") or {}
    user = data.get("user") or data
    return {
        "open_id": user.get("open_id") or data.get("open_id"),
        "display_name": user.get("display_name") or data.get("display_name"),
        "avatar_url": user.get("avatar_url") or data.get("avatar_url"),
        "profile_url": user.get("profile_deep_link") or data.get("profile_deep_link"),
        "username": user.get("username") or data.get("username"),
        "bio_description": user.get("bio_description") or data.get("bio_description"),
        "is_verified": bool(user.get("is_verified") or data.get("is_verified")),
    }


async def _fetch_creator_info(access_token: str) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            TIKTOK_CREATOR_INFO_URL,
            json={},
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=UTF-8",
            },
        )
    if resp.status_code != 200:
        _raise_from_tiktok_response(resp, "Erro ao consultar permissões de publicação do TikTok")

    payload = resp.json()
    error = payload.get("error") or {}
    if error.get("code") not in {None, "ok"}:
        raise TikTokApiError(error.get("message") or "Erro ao consultar permissões do TikTok", payload=payload)

    data = payload.get("data") or {}
    privacy_options = data.get("privacy_level_options") or DEFAULT_PRIVACY_OPTIONS
    return {
        "creator_username": data.get("creator_username"),
        "creator_nickname": data.get("creator_nickname"),
        "creator_avatar_url": data.get("creator_avatar_url"),
        "privacy_level_options": privacy_options,
        "comment_disabled": bool(data.get("comment_disabled", False)),
        "duet_disabled": bool(data.get("duet_disabled", False)),
        "stitch_disabled": bool(data.get("stitch_disabled", False)),
        "max_video_post_duration_sec": int(data.get("max_video_post_duration_sec") or 600),
    }


async def _get_post_status(access_token: str, publish_id: str) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            TIKTOK_STATUS_URL,
            json={"publish_id": publish_id},
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=UTF-8",
            },
        )
    if resp.status_code != 200:
        _raise_from_tiktok_response(resp, "Erro ao consultar status da publicação do TikTok")
    payload = resp.json()
    error = payload.get("error") or {}
    if error.get("code") not in {None, "ok"}:
        raise TikTokApiError(error.get("message") or "Erro ao consultar status da publicação do TikTok", payload=payload)
    return payload.get("data") or {}


def _update_user_from_tiktok(current_user: User, profile: dict, creator: Optional[dict], *, token_data: Optional[dict] = None) -> None:
    if token_data:
        current_user.tiktok_open_id = token_data.get("open_id") or current_user.tiktok_open_id
        current_user.tiktok_scope = _normalize_scope_value(token_data.get("scope") or current_user.tiktok_scope)
    current_user.tiktok_display_name = profile.get("display_name") or current_user.tiktok_display_name
    current_user.tiktok_username = profile.get("username") or creator.get("creator_username") if creator else profile.get("username")
    current_user.tiktok_avatar_url = profile.get("avatar_url") or (creator.get("creator_avatar_url") if creator else current_user.tiktok_avatar_url)
    current_user.tiktok_profile_url = profile.get("profile_url") or current_user.tiktok_profile_url
    current_user.tiktok_is_verified = bool(profile.get("is_verified"))
    if creator:
        current_user.tiktok_privacy_options_json = ",".join(creator.get("privacy_level_options") or [])


@router.get("/auth-url")
def get_auth_url(
    state: Optional[str] = None,
    code_challenge: Optional[str] = None,
    current_user: User = Depends(get_current_user),
):
    _ensure_configured()
    final_state = state or secrets.token_urlsafe(24)
    final_code_challenge = str(code_challenge or "").strip()
    if not final_code_challenge:
        raise HTTPException(status_code=400, detail="code_challenge não informado para o OAuth do TikTok.")

    params = httpx.QueryParams(
        {
            "client_key": TIKTOK_CLIENT_KEY,
            "scope": ",".join(TIKTOK_CONNECT_SCOPES),
            "response_type": "code",
            "redirect_uri": TIKTOK_REDIRECT_URI,
            "state": final_state,
            "code_challenge": final_code_challenge,
            "code_challenge_method": "S256",
        }
    )
    return {"url": f"{TIKTOK_AUTH_URL}?{params}", "state": final_state}


@router.post("/connect")
async def connect_tiktok(payload: dict, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    _ensure_configured()
    code = str(payload.get("code") or "").strip()
    code_verifier = str(payload.get("code_verifier") or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="Código de autorização do TikTok não informado.")
    if not code_verifier:
        raise HTTPException(status_code=400, detail="code_verifier não informado para concluir o OAuth do TikTok.")

    token_data = await _post_form(
        TIKTOK_TOKEN_URL,
        {
            "client_key": TIKTOK_CLIENT_KEY,
            "client_secret": TIKTOK_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": TIKTOK_REDIRECT_URI,
            "code_verifier": code_verifier,
        },
    )

    granted_scopes = _scope_set(token_data.get("scope"))
    missing = [scope for scope in ("video.publish", "user.info.profile") if scope not in granted_scopes]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "A autorização do TikTok voltou sem os escopos necessários. "
                f"Faltando: {', '.join(missing)}. Revise o app no TikTok Developers e reconecte a conta."
            ),
        )

    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")
    expires_in = int(token_data.get("expires_in", 86400))
    refresh_expires_in = int(token_data.get("refresh_expires_in", 31536000))
    if not access_token or not refresh_token:
        raise HTTPException(status_code=400, detail="O TikTok não retornou os tokens esperados.")

    current_user.tiktok_access_token = access_token
    current_user.tiktok_refresh_token = refresh_token
    current_user.tiktok_token_expires_at = _utcnow() + timedelta(seconds=expires_in)
    current_user.tiktok_refresh_token_expires_at = _utcnow() + timedelta(seconds=refresh_expires_in)
    current_user.tiktok_open_id = token_data.get("open_id")
    current_user.tiktok_scope = _normalize_scope_value(token_data.get("scope"))

    profile = await _fetch_user_profile(access_token)
    creator = await _fetch_creator_info(access_token)
    _update_user_from_tiktok(current_user, profile, creator, token_data=token_data)

    session.add(current_user)
    session.commit()
    session.refresh(current_user)

    return {
        "ok": True,
        "message": "Conta do TikTok conectada com sucesso!",
        "display_name": current_user.tiktok_display_name,
        "username": current_user.tiktok_username,
        "avatar_url": current_user.tiktok_avatar_url,
        "profile_url": current_user.tiktok_profile_url,
        "privacy_level_options": creator.get("privacy_level_options") or DEFAULT_PRIVACY_OPTIONS,
        "max_video_post_duration_sec": creator.get("max_video_post_duration_sec") or 600,
    }


@router.get("/status")
async def tiktok_status(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    connected = bool(current_user.tiktok_refresh_token and current_user.tiktok_open_id)
    privacy_options = [item for item in (current_user.tiktok_privacy_options_json or "").split(",") if item]
    max_video_post_duration_sec = 600
    if connected:
        try:
            access_token = await _refresh_access_token_if_needed(current_user, session)
            profile = await _fetch_user_profile(access_token)
            creator = await _fetch_creator_info(access_token)
            _update_user_from_tiktok(current_user, profile, creator)
            privacy_options = creator.get("privacy_level_options") or DEFAULT_PRIVACY_OPTIONS
            max_video_post_duration_sec = creator.get("max_video_post_duration_sec") or 600
            session.add(current_user)
            session.commit()
            session.refresh(current_user)
        except Exception:
            pass

    return {
        "connected": connected,
        "display_name": current_user.tiktok_display_name,
        "username": current_user.tiktok_username,
        "avatar_url": current_user.tiktok_avatar_url,
        "profile_url": current_user.tiktok_profile_url,
        "is_verified": bool(current_user.tiktok_is_verified),
        "privacy_level_options": privacy_options or DEFAULT_PRIVACY_OPTIONS,
        "privacy_level_labels": PRIVACY_LABELS,
        "max_video_post_duration_sec": max_video_post_duration_sec,
    }


@router.post("/disconnect")
async def disconnect_tiktok(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    if current_user.tiktok_access_token:
        try:
            await _post_form(
                TIKTOK_REVOKE_URL,
                {
                    "client_key": TIKTOK_CLIENT_KEY,
                    "client_secret": TIKTOK_CLIENT_SECRET,
                    "token": current_user.tiktok_access_token,
                },
            )
        except Exception:
            pass

    current_user.tiktok_access_token = None
    current_user.tiktok_refresh_token = None
    current_user.tiktok_token_expires_at = None
    current_user.tiktok_refresh_token_expires_at = None
    current_user.tiktok_open_id = None
    current_user.tiktok_scope = None
    current_user.tiktok_display_name = None
    current_user.tiktok_username = None
    current_user.tiktok_avatar_url = None
    current_user.tiktok_profile_url = None
    current_user.tiktok_is_verified = False
    current_user.tiktok_privacy_options_json = None
    session.add(current_user)
    session.commit()
    return {"ok": True}


@router.post("/publish")
async def publish_tiktok(
    caption: str = Form(""),
    privacy_level: str = Form("SELF_ONLY"),
    disable_comment: bool = Form(False),
    disable_duet: bool = Form(False),
    disable_stitch: bool = Form(False),
    is_aigc: bool = Form(False),
    brand_content_toggle: bool = Form(False),
    brand_organic_toggle: bool = Form(False),
    video_cover_timestamp_ms: int = Form(1000),
    video_file: UploadFile = File(...),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    if not video_file.filename:
        raise HTTPException(status_code=400, detail="Selecione um vídeo para publicar no TikTok.")

    access_token = await _refresh_access_token_if_needed(current_user, session)
    creator = await _fetch_creator_info(access_token)
    privacy_options = creator.get("privacy_level_options") or DEFAULT_PRIVACY_OPTIONS
    if privacy_level not in privacy_options:
        raise HTTPException(
            status_code=400,
            detail=f"A privacidade escolhida não é permitida para essa conta. Opções atuais: {', '.join(privacy_options)}",
        )

    content = await video_file.read()
    if not content:
        raise HTTPException(status_code=400, detail="O arquivo de vídeo enviado está vazio.")

    video_size = len(content)
    chunk_size = min(max(5 * 1024 * 1024, video_size), 64 * 1024 * 1024)
    total_chunk_count = max(1, math.ceil(video_size / chunk_size))
    content_type = video_file.content_type or "video/mp4"

    async with httpx.AsyncClient(timeout=120.0) as client:
        init_resp = await client.post(
            TIKTOK_VIDEO_INIT_URL,
            json={
                "post_info": {
                    "title": (caption or "").strip(),
                    "privacy_level": privacy_level,
                    "disable_comment": disable_comment,
                    "disable_duet": disable_duet,
                    "disable_stitch": disable_stitch,
                    "video_cover_timestamp_ms": max(0, int(video_cover_timestamp_ms or 0)),
                    "brand_content_toggle": brand_content_toggle,
                    "brand_organic_toggle": brand_organic_toggle,
                    "is_aigc": is_aigc,
                },
                "source_info": {
                    "source": "FILE_UPLOAD",
                    "video_size": video_size,
                    "chunk_size": chunk_size,
                    "total_chunk_count": total_chunk_count,
                },
            },
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=UTF-8",
            },
        )

    if init_resp.status_code != 200:
        _raise_from_tiktok_response(init_resp, "Erro ao iniciar publicação no TikTok")

    init_payload = init_resp.json()
    init_error = init_payload.get("error") or {}
    if init_error.get("code") not in {None, "ok"}:
        raise HTTPException(status_code=400, detail=init_error.get("message") or "Erro ao iniciar publicação no TikTok")

    init_data = init_payload.get("data") or {}
    upload_url = init_data.get("upload_url")
    publish_id = init_data.get("publish_id")
    if not upload_url or not publish_id:
        raise HTTPException(status_code=400, detail="O TikTok não retornou upload_url/publish_id para a publicação.")

    async with httpx.AsyncClient(timeout=300.0) as client:
        for idx in range(total_chunk_count):
            start = idx * chunk_size
            end = min(start + chunk_size, video_size)
            chunk = content[start:end]
            headers = {
                "Content-Type": content_type,
                "Content-Length": str(len(chunk)),
                "Content-Range": f"bytes {start}-{end - 1}/{video_size}",
            }
            upload_resp = await client.put(upload_url, content=chunk, headers=headers)
            if upload_resp.status_code not in {200, 201, 204, 206}:
                detail = upload_resp.text or "Falha ao enviar o vídeo para o TikTok."
                raise HTTPException(status_code=400, detail=f"Erro no upload do vídeo para o TikTok: {detail}")

    status_data = {}
    status_warning = None
    try:
        status_data = await _get_post_status(access_token, publish_id)
    except Exception as exc:
        status_warning = str(exc)

    return {
        "ok": True,
        "message": "Vídeo enviado ao TikTok com sucesso.",
        "publish_id": publish_id,
        "status": status_data.get("status") or "PROCESSING_UPLOAD",
        "status_message": status_data.get("fail_reason") or status_data.get("status_message"),
        "warning": status_warning,
    }
