from __future__ import annotations

import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlmodel import Session

from .config import (
    GOOGLE_BUSINESS_CLIENT_ID,
    GOOGLE_BUSINESS_CLIENT_SECRET,
    GOOGLE_BUSINESS_REDIRECT_URI,
)
from .db import get_session
from .deps import get_current_user
from .models import User

router = APIRouter(prefix="/api/google-business-profile", tags=["google-business-profile"])

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_OAUTH_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_BUSINESS_SCOPE = "https://www.googleapis.com/auth/business.manage"
ACCOUNT_MANAGEMENT_BASE = "https://mybusinessaccountmanagement.googleapis.com/v1"
BUSINESS_INFO_BASE = "https://mybusinessbusinessinformation.googleapis.com/v1"
MYBUSINESS_V4_BASE = "https://mybusiness.googleapis.com/v4"




def _extract_google_error_payload(payload: Any) -> dict:
    if isinstance(payload, dict):
        return payload.get("error") if isinstance(payload.get("error"), dict) else payload
    return {}


def _google_api_http_exception(resp: httpx.Response, fallback: str) -> HTTPException:
    try:
        payload = resp.json()
    except Exception:
        payload = {"raw": resp.text}

    error = _extract_google_error_payload(payload)
    raw_message = str(error.get("message") or payload.get("message") or resp.text or fallback).strip()
    status_text = str(error.get("status") or "").strip().upper()
    reason = ""
    details = error.get("details")
    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict):
                reason = str(item.get("reason") or item.get("metadata", {}).get("quota_limit") or "").strip()
                if reason:
                    break

    message_lower = raw_message.lower()
    is_quota = (
        resp.status_code == 429
        or status_text in {"RESOURCE_EXHAUSTED", "RATE_LIMIT_EXCEEDED"}
        or reason in {"RATE_LIMIT_EXCEEDED", "DefaultRequestsPerMinutePerProject"}
        or "quota exceeded" in message_lower
        or "request a higher quota limit" in message_lower
        or 'quota_limit_value": "0"' in resp.text
    )

    if is_quota:
        detail = (
            "O Google aceitou a autenticação, mas a API do Perfil de Empresa no seu projeto está sem cota liberada. "
            "A quota atual de requisições por minuto está 0 no Google Cloud. "
            "Libere ou solicite aumento de quota para a API mybusinessaccountmanagement.googleapis.com e tente novamente."
        )
        return HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=detail)

    return HTTPException(status_code=400, detail=f"{fallback}: {raw_message}")


def _is_google_quota_exception(exc: HTTPException) -> bool:
    detail = str(exc.detail or "").lower()
    return exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS or "quota" in detail or "cota" in detail

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_configured() -> None:
    if not GOOGLE_BUSINESS_CLIENT_ID or not GOOGLE_BUSINESS_CLIENT_SECRET or not GOOGLE_BUSINESS_REDIRECT_URI:
        raise HTTPException(status_code=500, detail="Configuração do Google Business Profile OAuth incompleta no backend.")


def _token_is_expired(expires_at: Optional[datetime]) -> bool:
    if not expires_at:
        return True
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at <= (_utcnow() + timedelta(minutes=2))


class ConnectRequest(BaseModel):
    code: str


class ApplyServicesRequest(BaseModel):
    location_name: Optional[str] = None
    source_type: str = Field(..., pattern="^(keyword_list|service_cards)$")
    items: list[Any] = Field(default_factory=list)
    language_code: str = "pt-BR"


async def _exchange_code(code: str) -> dict:
    _ensure_configured()
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": GOOGLE_BUSINESS_CLIENT_ID,
                "client_secret": GOOGLE_BUSINESS_CLIENT_SECRET,
                "redirect_uri": GOOGLE_BUSINESS_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if resp.status_code != 200:
        raise _google_api_http_exception(resp, "Erro ao conectar Google Business Profile")
    return resp.json()


async def _refresh_access_token_if_needed(current_user: User, session: Session) -> str:
    _ensure_configured()
    if current_user.google_business_access_token and not _token_is_expired(current_user.google_business_token_expires_at):
        return current_user.google_business_access_token

    if not current_user.google_business_refresh_token:
        raise HTTPException(status_code=400, detail="Perfil de Empresa Google não está conectado nesta conta ou requer reconexão.")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": GOOGLE_BUSINESS_CLIENT_ID,
                "client_secret": GOOGLE_BUSINESS_CLIENT_SECRET,
                "refresh_token": current_user.google_business_refresh_token,
                "grant_type": "refresh_token",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if resp.status_code != 200:
        raise _google_api_http_exception(resp, "Erro ao renovar token do Perfil de Empresa Google")

    data = resp.json()
    access_token = data.get("access_token")
    expires_in = int(data.get("expires_in", 3600))
    if not access_token:
        raise HTTPException(status_code=400, detail="O Google não retornou um access_token válido para o Perfil de Empresa.")

    current_user.google_business_access_token = access_token
    current_user.google_business_token_expires_at = _utcnow() + timedelta(seconds=expires_in)
    session.add(current_user)
    session.commit()
    session.refresh(current_user)
    return access_token


async def _google_get(url: str, access_token: str, *, params: Optional[dict] = None) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(url, params=params, headers={"Authorization": f"Bearer {access_token}"})
    if resp.status_code != 200:
        raise _google_api_http_exception(resp, "Erro na API do Perfil de Empresa Google")
    return resp.json()


async def _google_patch(url: str, access_token: str, *, json_body: dict, params: Optional[dict] = None) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.patch(url, params=params, json=json_body, headers={"Authorization": f"Bearer {access_token}"})
    if resp.status_code != 200:
        raise _google_api_http_exception(resp, "Erro ao atualizar serviços do Perfil de Empresa Google")
    return resp.json()


def _location_title(location: dict) -> str:
    return str(location.get("title") or location.get("storeCode") or location.get("name") or "Local sem título")


def _location_category(location: dict) -> Optional[str]:
    categories = location.get("categories") or {}
    primary = categories.get("primaryCategory") or {}
    return primary.get("name")


async def _list_accounts(access_token: str) -> list[dict]:
    data = await _google_get(f"{ACCOUNT_MANAGEMENT_BASE}/accounts", access_token, params={"pageSize": 20})
    return data.get("accounts") or []


async def _list_locations_for_account(access_token: str, account_name: str) -> list[dict]:
    params = {
        "pageSize": 100,
        "readMask": "name,title,storeCode,categories,metadata,profile,languageCode",
    }
    data = await _google_get(f"{BUSINESS_INFO_BASE}/{account_name}/locations", access_token, params=params)
    return data.get("locations") or []


async def _get_v4_service_list(access_token: str, account_location_name: str) -> dict:
    return await _google_get(f"{MYBUSINESS_V4_BASE}/{account_location_name}/serviceList", access_token)


async def _hydrate_locations(access_token: str) -> tuple[Optional[dict], list[dict]]:
    accounts = await _list_accounts(access_token)
    if not accounts:
        return None, []

    chosen_account = None
    chosen_locations: list[dict] = []
    for account in accounts:
        account_name = account.get("name")
        if not account_name:
            continue
        locations = await _list_locations_for_account(access_token, account_name)
        if locations:
            chosen_account = account
            chosen_locations = locations
            break

    if not chosen_account:
        chosen_account = accounts[0]
        chosen_locations = []

    normalized_locations = []
    for loc in chosen_locations:
        normalized_locations.append(
            {
                "name": loc.get("name"),
                "title": _location_title(loc),
                "store_code": loc.get("storeCode"),
                "category": _location_category(loc),
                "language_code": loc.get("languageCode") or "pt-BR",
            }
        )
    return chosen_account, normalized_locations


@router.get("/auth-url")
def get_auth_url(current_user: User = Depends(get_current_user)):
    _ensure_configured()
    state = secrets.token_urlsafe(24)
    params = {
        "client_id": GOOGLE_BUSINESS_CLIENT_ID,
        "redirect_uri": GOOGLE_BUSINESS_REDIRECT_URI,
        "response_type": "code",
        "access_type": "offline",
        "include_granted_scopes": "true",
        "prompt": "consent",
        "scope": GOOGLE_BUSINESS_SCOPE,
        "state": state,
    }
    return {"url": f"{GOOGLE_OAUTH_AUTH_URL}?{httpx.QueryParams(params)}", "state": state}


@router.post("/connect")
async def connect_google_business(payload: ConnectRequest, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    token_data = await _exchange_code(payload.code)
    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token") or current_user.google_business_refresh_token
    expires_in = int(token_data.get("expires_in", 3600))
    if not access_token:
        raise HTTPException(status_code=400, detail="O Google não retornou um access_token válido.")

    current_user.google_business_access_token = access_token
    current_user.google_business_refresh_token = refresh_token
    current_user.google_business_token_expires_at = _utcnow() + timedelta(seconds=expires_in)
    session.add(current_user)
    session.commit()
    session.refresh(current_user)

    quota_warning = None
    account = None
    locations: list[dict] = []
    try:
        account, locations = await _hydrate_locations(access_token)
    except HTTPException as exc:
        if _is_google_quota_exception(exc):
            quota_warning = str(exc.detail)
        else:
            raise

    current_user.google_business_account_name = (account or {}).get("name") or current_user.google_business_account_name
    current_user.google_business_account_display_name = (account or {}).get("accountName") or current_user.google_business_account_display_name
    current_user.google_business_locations_json = json.dumps(locations, ensure_ascii=False) if locations else (current_user.google_business_locations_json or "[]")

    selected = locations[0] if locations else None
    if selected:
        current_user.google_business_location_name = selected.get("name")
        current_user.google_business_location_title = selected.get("title")
        current_user.google_business_location_store_code = selected.get("store_code")
        current_user.google_business_location_category = selected.get("category")

    session.add(current_user)
    session.commit()

    return {
        "ok": True,
        "message": "Perfil de Empresa Google conectado com sucesso!" if not quota_warning else "Perfil de Empresa Google autenticado, mas a API ainda está sem cota liberada no Google Cloud.",
        "warning": quota_warning,
        "account_name": current_user.google_business_account_display_name,
        "location_title": current_user.google_business_location_title,
        "locations": locations,
    }


@router.get("/status")
async def google_business_status(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    connected = bool(current_user.google_business_refresh_token)
    sync_warning = None
    locations = []
    try:
        locations = json.loads(current_user.google_business_locations_json or "[]")
    except Exception:
        locations = []

    if connected:
        try:
            access_token = await _refresh_access_token_if_needed(current_user, session)
            account, fresh_locations = await _hydrate_locations(access_token)
            if account:
                current_user.google_business_account_name = account.get("name")
                current_user.google_business_account_display_name = account.get("accountName")
            if fresh_locations:
                current_user.google_business_locations_json = json.dumps(fresh_locations, ensure_ascii=False)
                locations = fresh_locations
                selected_name = current_user.google_business_location_name or fresh_locations[0].get("name")
                selected = next((item for item in fresh_locations if item.get("name") == selected_name), fresh_locations[0])
                current_user.google_business_location_name = selected.get("name")
                current_user.google_business_location_title = selected.get("title")
                current_user.google_business_location_store_code = selected.get("store_code")
                current_user.google_business_location_category = selected.get("category")
                session.add(current_user)
                session.commit()
                session.refresh(current_user)
        except HTTPException as exc:
            if _is_google_quota_exception(exc):
                sync_warning = str(exc.detail)
            else:
                pass

    return {
        "connected": connected,
        "account_name": current_user.google_business_account_display_name,
        "location_name": current_user.google_business_location_name,
        "location_title": current_user.google_business_location_title,
        "location_store_code": current_user.google_business_location_store_code,
        "location_category": current_user.google_business_location_category,
        "locations": locations,
        "warning": sync_warning,
    }


@router.post("/disconnect")
def disconnect_google_business(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    current_user.google_business_access_token = None
    current_user.google_business_refresh_token = None
    current_user.google_business_token_expires_at = None
    current_user.google_business_account_name = None
    current_user.google_business_account_display_name = None
    current_user.google_business_location_name = None
    current_user.google_business_location_title = None
    current_user.google_business_location_store_code = None
    current_user.google_business_location_category = None
    current_user.google_business_locations_json = "[]"
    session.add(current_user)
    session.commit()
    return {"ok": True}


@router.get("/locations")
async def list_google_business_locations(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    await _refresh_access_token_if_needed(current_user, session)
    try:
        locations = json.loads(current_user.google_business_locations_json or "[]")
    except Exception:
        locations = []
    return {
        "locations": locations,
        "selected_location_name": current_user.google_business_location_name,
        "selected_location_title": current_user.google_business_location_title,
    }


@router.post("/locations/select")
def select_google_business_location(payload: dict, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    location_name = str(payload.get("location_name") or "").strip()
    if not location_name:
        raise HTTPException(status_code=400, detail="Local do Perfil de Empresa Google não informado.")
    try:
        locations = json.loads(current_user.google_business_locations_json or "[]")
    except Exception:
        locations = []
    selected = next((item for item in locations if item.get("name") == location_name), None)
    if not selected:
        raise HTTPException(status_code=404, detail="Local não encontrado na lista conectada.")

    current_user.google_business_location_name = selected.get("name")
    current_user.google_business_location_title = selected.get("title")
    current_user.google_business_location_store_code = selected.get("store_code")
    current_user.google_business_location_category = selected.get("category")
    session.add(current_user)
    session.commit()
    return {"ok": True, "location_title": current_user.google_business_location_title}


def _sanitize_text(value: Any, max_len: int) -> str:
    text = str(value or "").strip()
    text = " ".join(text.split())
    return text[:max_len].strip()


def _dedupe_keep_order(items: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for item in items:
        key = (
            item.get("label", {}).get("displayName", "").strip().lower(),
            item.get("label", {}).get("description", "").strip().lower(),
        )
        if not key[0] or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _build_service_items(source_type: str, raw_items: list[Any], category: str, language_code: str) -> list[dict]:
    built: list[dict] = []

    if source_type == "keyword_list":
        for raw in raw_items:
            label = _sanitize_text(raw, 140)
            if not label:
                continue
            built.append(
                {
                    "isOffered": True,
                    "freeFormServiceItem": {
                        "category": category,
                        "label": {
                            "displayName": label,
                            "languageCode": language_code,
                        },
                    },
                }
            )
    elif source_type == "service_cards":
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            name = _sanitize_text(raw.get("nome"), 140)
            description = _sanitize_text(raw.get("descricao"), 250)
            if name:
                built.append(
                    {
                        "isOffered": True,
                        "freeFormServiceItem": {
                            "category": category,
                            "label": {
                                "displayName": name,
                                "description": description,
                                "languageCode": language_code,
                            },
                        },
                    }
                )
            keywords = raw.get("palavras_chave") or []
            if isinstance(keywords, list):
                for keyword in keywords:
                    label = _sanitize_text(keyword, 140)
                    if not label:
                        continue
                    built.append(
                        {
                            "isOffered": True,
                            "freeFormServiceItem": {
                                "category": category,
                                "label": {
                                    "displayName": label,
                                    "languageCode": language_code,
                                },
                            },
                        }
                    )
    else:
        raise HTTPException(status_code=400, detail="source_type inválido para aplicação de serviços.")

    normalized = []
    for item in built:
        ff = item.get("freeFormServiceItem") or {}
        label = ff.get("label") or {}
        normalized.append(
            {
                "isOffered": True,
                "freeFormServiceItem": {
                    "category": category,
                    "label": {
                        "displayName": _sanitize_text(label.get("displayName"), 140),
                        **({"description": _sanitize_text(label.get("description"), 250)} if _sanitize_text(label.get("description"), 250) else {}),
                        "languageCode": _sanitize_text(label.get("languageCode") or language_code, 20) or "pt-BR",
                    },
                },
            }
        )
    return _dedupe_keep_order(normalized)


@router.post("/services/apply")
async def apply_google_business_services(payload: ApplyServicesRequest, session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    access_token = await _refresh_access_token_if_needed(current_user, session)
    location_name = payload.location_name or current_user.google_business_location_name
    if not location_name:
        raise HTTPException(status_code=400, detail="Nenhum local do Perfil de Empresa Google foi selecionado.")

    try:
        locations = json.loads(current_user.google_business_locations_json or "[]")
    except Exception:
        locations = []
    selected = next((item for item in locations if item.get("name") == location_name), None)
    if not selected:
        raise HTTPException(status_code=404, detail="Local selecionado não foi encontrado.")

    category = selected.get("category") or current_user.google_business_location_category
    if not category:
        raise HTTPException(status_code=400, detail="Não foi possível identificar a categoria principal do local para montar a lista de serviços.")

    account_location_name = location_name.replace("locations/", "accounts/me/locations/") if location_name.startswith("locations/") else location_name
    service_list_name = f"{account_location_name}/serviceList"
    language_code = payload.language_code or "pt-BR"
    service_items = _build_service_items(payload.source_type, payload.items, category, language_code)
    if not service_items:
        raise HTTPException(status_code=400, detail="Nenhum item válido foi encontrado para aplicar no Perfil de Empresa Google.")

    request_body = {
        "name": service_list_name,
        "serviceItems": service_items,
    }

    updated = await _google_patch(
        f"{MYBUSINESS_V4_BASE}/{account_location_name}/serviceList",
        access_token,
        json_body=request_body,
        params={"updateMask": "serviceItems"},
    )

    current_user.google_business_location_name = selected.get("name")
    current_user.google_business_location_title = selected.get("title")
    current_user.google_business_location_store_code = selected.get("store_code")
    current_user.google_business_location_category = selected.get("category")
    session.add(current_user)
    session.commit()

    return {
        "ok": True,
        "message": "Serviços do Perfil de Empresa Google atualizados com sucesso.",
        "applied_count": len(updated.get("serviceItems") or service_items),
        "location_title": selected.get("title"),
        "service_list": updated,
    }


@router.get("/services/current")
async def get_current_google_business_services(session: Session = Depends(get_session), current_user: User = Depends(get_current_user)):
    access_token = await _refresh_access_token_if_needed(current_user, session)
    location_name = current_user.google_business_location_name
    if not location_name:
        raise HTTPException(status_code=400, detail="Nenhum local do Perfil de Empresa Google foi selecionado.")
    account_location_name = location_name.replace("locations/", "accounts/me/locations/") if location_name.startswith("locations/") else location_name
    data = await _get_v4_service_list(access_token, account_location_name)
    return data
