from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlmodel import Session

from .config import INSTAGRAM_META_APP_ID, INSTAGRAM_META_APP_SECRET, META_GRAPH_VERSION
from .db import get_session
from .deps import get_current_user
from .models import User

router = APIRouter(prefix="/api/instagram", tags=["instagram"])
GRAPH_VERSION = META_GRAPH_VERSION or "v19.0"
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_VERSION}"


class LinkRequest(BaseModel):
    code: str
    redirect_uri: str


class PublishRequest(BaseModel):
    caption: str = Field(default="", max_length=2200)
    image_url: Optional[str] = None
    carousel_images: list[str] = Field(default_factory=list)
    collaborators: list[str] = Field(default_factory=list)
    location_id: Optional[str] = None
    first_comment: Optional[str] = None
    share_to_feed: Optional[bool] = None

    @model_validator(mode="after")
    def validate_media(self):
        has_single = bool((self.image_url or "").strip())
        carousel = [u for u in self.carousel_images if str(u).strip()]
        if not has_single and not carousel:
            raise ValueError("Informe uma image_url ou pelo menos uma imagem no carrossel.")
        return self


async def _raise_if_graph_error(response: httpx.Response, fallback: str):
    try:
        data = response.json()
    except Exception:
        data = {"message": response.text or fallback}

    if response.status_code >= 400 or (isinstance(data, dict) and "error" in data):
        error = data.get("error") if isinstance(data, dict) else None
        raise HTTPException(status_code=400, detail=error or data or fallback)
    return data


async def _exchange_code_for_short_token(
    client: httpx.AsyncClient,
    code: str,
    redirect_uri: str,
    client_id: str = INSTAGRAM_META_APP_ID,
    client_secret: str = INSTAGRAM_META_APP_SECRET,
) -> str:
    token_res = await client.get(
        f"{GRAPH_BASE}/oauth/access_token",
        params={
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "code": code,
        },
    )
    token_data = await _raise_if_graph_error(token_res, "Falha ao obter token da Meta.")
    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="A Meta não devolveu access_token.")
    return access_token


async def _exchange_for_long_lived_token(
    client: httpx.AsyncClient,
    short_token: str,
    client_id: str = INSTAGRAM_META_APP_ID,
    client_secret: str = INSTAGRAM_META_APP_SECRET,
) -> tuple[str, Optional[int]]:
    long_res = await client.get(
        f"{GRAPH_BASE}/oauth/access_token",
        params={
            "grant_type": "fb_exchange_token",
            "client_id": client_id,
            "client_secret": client_secret,
            "fb_exchange_token": short_token,
        },
    )
    long_data = await _raise_if_graph_error(long_res, "Falha ao gerar token de longa duração.")
    return long_data.get("access_token", short_token), long_data.get("expires_in")


async def _get_pages(client: httpx.AsyncClient, access_token: str) -> list[dict]:
    pages_res = await client.get(
        f"{GRAPH_BASE}/me/accounts",
        params={
            "fields": "id,name,access_token",
            "access_token": access_token,
        },
    )
    pages_data = await _raise_if_graph_error(pages_res, "Falha ao obter páginas do Facebook.")
    return pages_data.get("data", [])


async def _find_instagram_account(client: httpx.AsyncClient, pages: list[dict], access_token: str) -> tuple[str, Optional[str], Optional[str]]:
    for page in pages:
        page_id = page.get("id")
        if not page_id:
            continue
        ig_res = await client.get(
            f"{GRAPH_BASE}/{page_id}",
            params={
                "fields": "instagram_business_account{id,username}",
                "access_token": access_token,
            },
        )
        ig_data = await _raise_if_graph_error(ig_res, "Falha ao obter Instagram vinculado à página.")
        ig_account = ig_data.get("instagram_business_account")
        if ig_account and ig_account.get("id"):
            return ig_account["id"], page_id, ig_account.get("username")
    raise HTTPException(
        status_code=400,
        detail="Nenhuma conta de Instagram profissional vinculada às páginas retornadas por /me/accounts.",
    )


def _stringify_graph_payload(payload: dict) -> dict:
    out: dict[str, str] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = "true" if value else "false"
        elif isinstance(value, (list, dict)):
            out[key] = json.dumps(value, ensure_ascii=False)
        else:
            out[key] = str(value)
    return out


async def _create_media_container(client: httpx.AsyncClient, ig_user_id: str, access_token: str, payload: dict) -> str:
    final_payload = _stringify_graph_payload(payload)
    print("PAYLOAD ENVIADO PRA META:", final_payload)
    res = await client.post(
        f"{GRAPH_BASE}/{ig_user_id}/media",
        data={**final_payload, "access_token": access_token},
    )
    data = await _raise_if_graph_error(res, "Erro ao criar contêiner de mídia no Instagram.")
    creation_id = data.get("id")
    if not creation_id:
        raise HTTPException(status_code=400, detail="A Meta não devolveu o ID do contêiner.")
    return creation_id


async def _wait_until_ready(
    client: httpx.AsyncClient,
    creation_id: str,
    access_token: str,
    timeout_seconds: int = 90,
    poll_seconds: int = 2,
) -> None:
    elapsed = 0
    while elapsed < timeout_seconds:
        res = await client.get(
            f"{GRAPH_BASE}/{creation_id}",
            params={
                "fields": "status_code,status",
                "access_token": access_token,
            },
        )
        data = await _raise_if_graph_error(res, "Erro ao consultar o status da mídia.")
        status_code = str(data.get("status_code") or "").upper()
        if status_code == "FINISHED":
            return
        if status_code == "ERROR":
            raise HTTPException(status_code=400, detail=data)
        await asyncio.sleep(poll_seconds)
        elapsed += poll_seconds

    raise HTTPException(
        status_code=400,
        detail="A mídia demorou mais do que o esperado para ficar pronta. Tente novamente em alguns segundos.",
    )


async def _publish_container(client: httpx.AsyncClient, ig_user_id: str, access_token: str, creation_id: str) -> str:
    last_error: Optional[HTTPException] = None
    for attempt in range(4):
        try:
            res = await client.post(
                f"{GRAPH_BASE}/{ig_user_id}/media_publish",
                data={"creation_id": creation_id, "access_token": access_token},
            )
            data = await _raise_if_graph_error(res, "Erro ao publicar no Instagram.")
            media_id = data.get("id")
            if not media_id:
                raise HTTPException(status_code=400, detail="A Meta não devolveu o ID da publicação.")
            return media_id
        except HTTPException as exc:
            last_error = exc
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            error = detail.get("error_user_title") or detail.get("message") or detail.get("error_user_msg") or ""
            code = detail.get("code")
            subcode = detail.get("error_subcode")
            ready_issue = code == 9007 or subcode == 2207027 or "not ready" in str(error).lower() or "não está pronta" in str(error).lower()
            if not ready_issue or attempt == 3:
                raise
            await asyncio.sleep(2 + attempt)
    if last_error:
        raise last_error
    raise HTTPException(status_code=400, detail="Erro inesperado ao publicar no Instagram.")


async def _create_first_comment(client: httpx.AsyncClient, media_id: str, access_token: str, text: str) -> None:
    if not text.strip():
        return
    res = await client.post(
        f"{GRAPH_BASE}/{media_id}/comments",
        data={"message": text, "access_token": access_token},
    )
    await _raise_if_graph_error(res, "A publicação saiu, mas houve erro ao criar o primeiro comentário.")


@router.get("/status")
async def instagram_status(current_user: User = Depends(get_current_user)):
    return {
        "connected": bool(current_user.instagram_account_id and current_user.instagram_meta_access_token),
        "instagram_account_id": current_user.instagram_account_id,
        "instagram_page_id": current_user.instagram_page_id,
        "instagram_username": current_user.instagram_username,
    }


@router.post("/disconnect")
async def instagram_disconnect(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    current_user.instagram_meta_access_token = None
    current_user.instagram_meta_token_expires_at = None
    current_user.instagram_account_id = None
    current_user.instagram_page_id = None
    current_user.instagram_username = None
    session.add(current_user)
    session.commit()
    return {"ok": True}


@router.post("/link")
async def link_instagram_account(
    req: LinkRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    if not INSTAGRAM_META_APP_ID or not INSTAGRAM_META_APP_SECRET:
        raise HTTPException(status_code=500, detail="INSTAGRAM_META_APP_ID e INSTAGRAM_META_APP_SECRET precisam estar configurados.")

    async with httpx.AsyncClient(timeout=40.0) as client:
        short_token = await _exchange_code_for_short_token(client, req.code, req.redirect_uri, INSTAGRAM_META_APP_ID, INSTAGRAM_META_APP_SECRET)
        access_token, expires_in = await _exchange_for_long_lived_token(client, short_token, INSTAGRAM_META_APP_ID, INSTAGRAM_META_APP_SECRET)
        pages = await _get_pages(client, access_token)
        instagram_account_id, instagram_page_id, instagram_username = await _find_instagram_account(client, pages, access_token)

    current_user.instagram_meta_access_token = access_token
    current_user.instagram_meta_token_expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=int(expires_in)) if expires_in else None
    )
    current_user.instagram_account_id = instagram_account_id
    current_user.instagram_page_id = instagram_page_id
    current_user.instagram_username = instagram_username
    session.add(current_user)
    session.commit()

    return {
        "ok": True,
        "message": "Conta vinculada com sucesso.",
        "instagram_username": instagram_username,
        "instagram_account_id": instagram_account_id,
        "instagram_page_id": instagram_page_id,
    }


@router.post("/publish")
async def publish_to_instagram(req: PublishRequest, current_user: User = Depends(get_current_user)):
    if not current_user.instagram_account_id or not current_user.instagram_meta_access_token:
        raise HTTPException(status_code=400, detail="Instagram não está vinculado nesta conta.")

    ig_user_id = current_user.instagram_account_id
    access_token = current_user.instagram_meta_access_token

    collaborators = [item.strip().lstrip("@") for item in req.collaborators if str(item).strip()]
    carousel_images = [item.strip() for item in req.carousel_images if str(item).strip()]

    async with httpx.AsyncClient(timeout=90.0) as client:
        if len(carousel_images) > 1:
            children_ids: list[str] = []
            for image_url in carousel_images:
                child_id = await _create_media_container(
                    client,
                    ig_user_id,
                    access_token,
                    {"image_url": image_url, "is_carousel_item": True},
                )
                await _wait_until_ready(client, child_id, access_token)
                children_ids.append(child_id)

            parent_payload: dict[str, object] = {
                "media_type": "CAROUSEL",
                "children": ",".join(children_ids),
                "caption": req.caption,
            }
            if req.location_id:
                parent_payload["location_id"] = req.location_id
            if collaborators:
                parent_payload["collaborators"] = collaborators

            creation_id = await _create_media_container(client, ig_user_id, access_token, parent_payload)
            await _wait_until_ready(client, creation_id, access_token)
        else:
            image_url = carousel_images[0] if carousel_images else (req.image_url or "").strip()
            payload: dict[str, object] = {
                "image_url": image_url,
                "caption": req.caption,
            }
            if req.location_id:
                payload["location_id"] = req.location_id
            if collaborators:
                payload["collaborators"] = collaborators
            creation_id = await _create_media_container(client, ig_user_id, access_token, payload)
            await _wait_until_ready(client, creation_id, access_token)

        media_id = await _publish_container(client, ig_user_id, access_token, creation_id)

        comment_warning = None
        if req.first_comment and req.first_comment.strip():
            try:
                await _create_first_comment(client, media_id, access_token, req.first_comment.strip())
            except HTTPException as exc:
                comment_warning = exc.detail

    return {
        "ok": True,
        "message": "Publicado com sucesso!",
        "post_id": media_id,
        "instagram_username": current_user.instagram_username,
        "warning": comment_warning,
    }
