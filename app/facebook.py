from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator
from sqlmodel import Session

from .config import FACEBOOK_META_APP_ID, FACEBOOK_META_APP_SECRET
from .db import get_session
from .deps import get_current_user
from .models import User
from .instagram import (
    GRAPH_BASE,
    _exchange_code_for_short_token,
    _exchange_for_long_lived_token,
    _get_pages,
    _raise_if_graph_error,
)

router = APIRouter(prefix="/api/facebook", tags=["facebook"])


class LinkRequest(BaseModel):
    code: str
    redirect_uri: str


class SelectPageRequest(BaseModel):
    page_id: str


class PublishRequest(BaseModel):
    message: str = Field(default="", max_length=63206)
    link: Optional[str] = None
    image_url: Optional[str] = None
    carousel_images: list[str] = Field(default_factory=list)
    published: bool = True
    scheduled_publish_time: Optional[int] = None
    backdated_time: Optional[str] = None
    place: Optional[str] = None
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_payload(self):
        has_message = bool(self.message.strip())
        has_link = bool((self.link or "").strip())
        has_image = bool((self.image_url or "").strip())
        has_carousel = any(str(item).strip() for item in self.carousel_images)
        if not any([has_message, has_link, has_image, has_carousel]):
            raise ValueError("Informe pelo menos texto, link, imagem ou carrossel para publicar no Facebook.")
        if self.scheduled_publish_time and self.published:
            raise ValueError("Para agendar uma publicação, marque como não publicada.")
        return self


async def _load_pages_with_details(client: httpx.AsyncClient, access_token: str) -> list[dict]:
    pages = await _get_pages(client, access_token)
    enriched: list[dict] = []
    for page in pages:
        page_id = page.get("id")
        page_token = page.get("access_token")
        if not page_id or not page_token:
            continue
        res = await client.get(
            f"{GRAPH_BASE}/{page_id}",
            params={
                "fields": "id,name,username,fan_count,followers_count,picture{url}",
                "access_token": access_token,
            },
        )
        data = await _raise_if_graph_error(res, "Falha ao obter detalhes da página do Facebook.")
        enriched.append(
            {
                "id": data.get("id") or page_id,
                "name": data.get("name") or page.get("name") or "Página sem nome",
                "username": data.get("username"),
                "fan_count": data.get("fan_count"),
                "followers_count": data.get("followers_count"),
                "picture_url": ((data.get("picture") or {}).get("data") or {}).get("url"),
                "access_token": page_token,
            }
        )
    return enriched


@router.get("/status")
async def facebook_status(current_user: User = Depends(get_current_user)):
    return {
        "connected": bool(current_user.facebook_page_id and current_user.facebook_page_access_token),
        "page_id": current_user.facebook_page_id,
        "page_name": current_user.facebook_page_name,
        "page_username": current_user.facebook_page_username,
        "pages": current_user.facebook_pages_json or "[]",
    }


@router.post("/disconnect")
async def facebook_disconnect(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    current_user.facebook_user_access_token = None
    current_user.facebook_user_token_expires_at = None
    current_user.facebook_page_id = None
    current_user.facebook_page_name = None
    current_user.facebook_page_username = None
    current_user.facebook_page_access_token = None
    current_user.facebook_pages_json = "[]"
    session.add(current_user)
    session.commit()
    return {"ok": True}


@router.post("/link")
async def link_facebook_account(
    req: LinkRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    if not FACEBOOK_META_APP_ID or not FACEBOOK_META_APP_SECRET:
        raise HTTPException(status_code=500, detail="FACEBOOK_META_APP_ID e FACEBOOK_META_APP_SECRET precisam estar configurados.")

    async with httpx.AsyncClient(timeout=40.0) as client:
        short_token = await _exchange_code_for_short_token(client, req.code, req.redirect_uri, FACEBOOK_META_APP_ID, FACEBOOK_META_APP_SECRET)
        access_token, expires_in = await _exchange_for_long_lived_token(client, short_token, FACEBOOK_META_APP_ID, FACEBOOK_META_APP_SECRET)
        pages = await _load_pages_with_details(client, access_token)

    if not pages:
        raise HTTPException(status_code=400, detail="Nenhuma página do Facebook foi encontrada para esta conta Meta.")

    selected_page = pages[0]

    current_user.facebook_user_access_token = access_token
    current_user.facebook_user_token_expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=int(expires_in)) if expires_in else None
    )
    current_user.facebook_page_id = selected_page.get("id")
    current_user.facebook_page_name = selected_page.get("name")
    current_user.facebook_page_username = selected_page.get("username")
    current_user.facebook_page_access_token = selected_page.get("access_token")
    current_user.facebook_pages_json = __import__("json").dumps(pages, ensure_ascii=False)
    session.add(current_user)
    session.commit()

    return {
        "ok": True,
        "message": "Página do Facebook vinculada com sucesso.",
        "page_id": current_user.facebook_page_id,
        "page_name": current_user.facebook_page_name,
        "page_username": current_user.facebook_page_username,
        "pages": pages,
    }


@router.post("/select-page")
async def facebook_select_page(
    req: SelectPageRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    import json

    try:
        pages = json.loads(current_user.facebook_pages_json or "[]")
    except Exception:
        pages = []

    selected = next((page for page in pages if str(page.get("id")) == str(req.page_id)), None)
    if not selected:
        raise HTTPException(status_code=404, detail="Página não encontrada entre as páginas autorizadas.")

    current_user.facebook_page_id = selected.get("id")
    current_user.facebook_page_name = selected.get("name")
    current_user.facebook_page_username = selected.get("username")
    current_user.facebook_page_access_token = selected.get("access_token")
    session.add(current_user)
    session.commit()

    return {"ok": True, "page_id": current_user.facebook_page_id, "page_name": current_user.facebook_page_name}


async def _create_photo(client: httpx.AsyncClient, page_id: str, page_token: str, image_url: str, published: bool = False) -> str:
    res = await client.post(
        f"{GRAPH_BASE}/{page_id}/photos",
        data={
            "url": image_url,
            "published": "true" if published else "false",
            "access_token": page_token,
        },
    )
    data = await _raise_if_graph_error(res, "Erro ao enviar imagem para o Facebook.")
    photo_id = data.get("id") or data.get("post_id")
    if not photo_id:
        raise HTTPException(status_code=400, detail="A Meta não devolveu o ID da foto do Facebook.")
    return str(photo_id)


@router.post("/publish")
async def publish_to_facebook(req: PublishRequest, current_user: User = Depends(get_current_user)):
    if not current_user.facebook_page_id or not current_user.facebook_page_access_token:
        raise HTTPException(status_code=400, detail="Facebook não está vinculado nesta conta.")

    page_id = current_user.facebook_page_id
    page_token = current_user.facebook_page_access_token
    carousel_images = [item.strip() for item in req.carousel_images if str(item).strip()]
    tags = [item.strip() for item in req.tags if str(item).strip()]

    feed_payload: dict[str, str] = {}
    if req.message.strip():
        feed_payload["message"] = req.message.strip()
    if req.link and req.link.strip():
        feed_payload["link"] = req.link.strip()
    if req.place and req.place.strip():
        feed_payload["place"] = req.place.strip()
    if tags:
        feed_payload["tags"] = ",".join(tags)
    if not req.published:
        feed_payload["published"] = "false"
    if req.scheduled_publish_time:
        feed_payload["scheduled_publish_time"] = str(req.scheduled_publish_time)
        feed_payload["unpublished_content_type"] = "SCHEDULED"
    if req.backdated_time and req.backdated_time.strip():
        feed_payload["backdated_time"] = req.backdated_time.strip()

    async with httpx.AsyncClient(timeout=90.0) as client:
        if len(carousel_images) > 1:
            attached_media: list[dict[str, str]] = []
            for image_url in carousel_images:
                media_fbid = await _create_photo(client, page_id, page_token, image_url, published=False)
                attached_media.append({"media_fbid": media_fbid})
            for idx, media in enumerate(attached_media):
                feed_payload[f"attached_media[{idx}]"] = __import__("json").dumps(media, ensure_ascii=False)
            res = await client.post(f"{GRAPH_BASE}/{page_id}/feed", data={**feed_payload, "access_token": page_token})
        elif req.image_url and req.image_url.strip():
            photo_payload = {k: v for k, v in feed_payload.items() if k not in {"link"}}
            photo_payload["url"] = req.image_url.strip()
            res = await client.post(f"{GRAPH_BASE}/{page_id}/photos", data={**photo_payload, "access_token": page_token})
        else:
            res = await client.post(f"{GRAPH_BASE}/{page_id}/feed", data={**feed_payload, "access_token": page_token})

        data = await _raise_if_graph_error(res, "Erro ao publicar no Facebook.")

    post_id = data.get("post_id") or data.get("id")
    if not post_id:
        raise HTTPException(status_code=400, detail="A Meta não devolveu o ID da publicação do Facebook.")

    return {
        "ok": True,
        "message": "Publicado com sucesso no Facebook!",
        "post_id": str(post_id),
        "page_id": page_id,
        "page_name": current_user.facebook_page_name,
    }
