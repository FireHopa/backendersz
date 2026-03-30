from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx
from PIL import Image, ImageOps, ImageChops, UnidentifiedImageError

from .image_local_edit import (
    analyze_all_regions_with_openai,
    analyze_region_with_openai,
    build_mask_from_analysis,
    extract_edit_instruction_info,
    render_all_local_text_replacements,
    render_local_text_fallback,
    should_use_local_text_render,
    should_use_localized_edit,
)
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from sqlmodel import Session, select

from .db import get_session
from .deps import get_current_user
from .models import ImageEngineProject, User


router = APIRouter()

OPENAI_CHAT_MODEL = "gpt-5.4"
OPENAI_IMAGE_MODEL = "gpt-image-1.5"

GEMINI_NATIVE_PRO_MODEL = "gemini-3-pro-image-preview"
GEMINI_NATIVE_FAST_MODEL = "gemini-3.1-flash-image-preview"
GOOGLE_IMAGEN_MODEL = "imagen-4.0-ultra-generate-001"

FAL_MODEL_PATH = "fal-ai/flux-pro/v1.1-ultra"

HTTP_TIMEOUT = httpx.Timeout(connect=20.0, read=180.0, write=60.0, pool=60.0)


class ImageEngineRequest(BaseModel):
    formato: str = Field(..., description="quadrado_1_1, vertical_9_16 ou horizontal_16_9")
    qualidade: str = Field(..., description="baixa, media ou alta")
    onde_postar: Optional[str] = Field(default=None, description="Campo legado ignorado")
    paleta_cores: str = Field(..., description="Paleta pronta ou personalizada")
    headline: str = ""
    subheadline: str = ""
    descricao_visual: str = ""
    width: Optional[int] = Field(default=None, description="Largura final customizada em pixels")
    height: Optional[int] = Field(default=None, description="Altura final customizada em pixels")

class ImageEditRequest(BaseModel):
    formato: str = Field(..., description="quadrado_1_1, vertical_9_16 ou horizontal_16_9")
    qualidade: str = Field(..., description="baixa, media ou alta")
    instrucoes_edicao: str = ""
    width: Optional[int] = Field(default=None, description="Largura final customizada em pixels")
    height: Optional[int] = Field(default=None, description="Altura final customizada em pixels")





class ImageEngineProjectPayload(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    position: int = Field(default=0, ge=0)
    is_current: bool = False
    snapshot: Dict[str, Any] = Field(default_factory=dict)


class ImageEngineProjectOut(BaseModel):
    id: str
    name: str
    position: int
    is_current: bool
    snapshot: Dict[str, Any]
    updated_at: str


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load_snapshot_json(raw_value: Optional[str]) -> Dict[str, Any]:
    if not raw_value:
        return {}
    try:
        data = json.loads(raw_value)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _serialize_image_project(project: ImageEngineProject) -> ImageEngineProjectOut:
    return ImageEngineProjectOut(
        id=project.public_id,
        name=project.name,
        position=project.position,
        is_current=project.is_current,
        snapshot=_load_snapshot_json(project.snapshot_json),
        updated_at=project.updated_at.isoformat() if project.updated_at else _utcnow().isoformat(),
    )


def _get_user_image_project_or_404(
    session: Session,
    user_id: int,
    public_id: str,
) -> ImageEngineProject:
    project = session.exec(
        select(ImageEngineProject).where(
            ImageEngineProject.user_id == user_id,
            ImageEngineProject.public_id == public_id,
        )
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projeto não encontrado.")
    return project


@router.get("/api/image-engine/projects")
def list_image_engine_projects(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    projects = session.exec(
        select(ImageEngineProject)
        .where(ImageEngineProject.user_id == current_user.id)
        .order_by(ImageEngineProject.position.asc(), ImageEngineProject.updated_at.desc())
    ).all()
    return {"projects": [_serialize_image_project(project).model_dump() for project in projects]}


@router.post("/api/image-engine/projects")
def create_image_engine_project(
    payload: ImageEngineProjectPayload,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    now = _utcnow()

    if payload.is_current:
        current_projects = session.exec(
            select(ImageEngineProject).where(ImageEngineProject.user_id == current_user.id)
        ).all()
        for item in current_projects:
            item.is_current = False
            item.updated_at = now
            session.add(item)

    project = ImageEngineProject(
        user_id=current_user.id,
        public_id=f"image-project-{os.urandom(8).hex()}",
        name=payload.name.strip(),
        position=payload.position,
        snapshot_json=json.dumps(payload.snapshot or {}, ensure_ascii=False),
        is_current=payload.is_current,
        created_at=now,
        updated_at=now,
    )
    session.add(project)
    session.commit()
    session.refresh(project)
    return {"project": _serialize_image_project(project).model_dump()}


@router.put("/api/image-engine/projects/{public_id}")
def update_image_engine_project(
    public_id: str,
    payload: ImageEngineProjectPayload,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    now = _utcnow()
    project = _get_user_image_project_or_404(session, current_user.id, public_id)

    if payload.is_current:
        current_projects = session.exec(
            select(ImageEngineProject).where(ImageEngineProject.user_id == current_user.id)
        ).all()
        for item in current_projects:
            item.is_current = False
            item.updated_at = now
            session.add(item)

    project.name = payload.name.strip()
    project.position = payload.position
    project.snapshot_json = json.dumps(payload.snapshot or {}, ensure_ascii=False)
    project.is_current = payload.is_current
    project.updated_at = now

    session.add(project)
    session.commit()
    session.refresh(project)
    return {"project": _serialize_image_project(project).model_dump()}




SUPPORTED_BASE_SIZES: List[Tuple[int, int]] = [
    (1024, 1024),
    (1024, 1536),
    (1536, 1024),
]


def _normalize_dimension_value(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        raise ValueError("Width e height precisam ser números inteiros.")

    if normalized < 256 or normalized > 4096:
        raise ValueError("Width e height precisam estar entre 256 e 4096 pixels.")

    return normalized


def _resolve_target_dimensions(width: Optional[int], height: Optional[int]) -> Optional[Tuple[int, int]]:
    normalized_width = _normalize_dimension_value(width)
    normalized_height = _normalize_dimension_value(height)

    if normalized_width is None and normalized_height is None:
        return None

    if normalized_width is None or normalized_height is None:
        raise ValueError("Para usar tamanho customizado, informe width e height.")

    return normalized_width, normalized_height


def _base_size_to_aspect_ratio(width: int, height: int) -> str:
    if width == height:
        return "1:1"
    if height > width:
        return "9:16"
    return "16:9"


def _choose_best_supported_base_size(target_width: int, target_height: int) -> Tuple[int, int]:
    best_size: Optional[Tuple[int, int]] = None
    best_score: Optional[Tuple[float, float, int]] = None

    for base_width, base_height in SUPPORTED_BASE_SIZES:
        scale = max(target_width / base_width, target_height / base_height)
        scaled_width = base_width * scale
        scaled_height = base_height * scale
        waste = (scaled_width * scaled_height) - (target_width * target_height)
        orientation_penalty = 0
        if (target_height > target_width and base_height < base_width) or (target_width > target_height and base_width < base_height):
            orientation_penalty = 1
        score = (orientation_penalty, waste, abs((base_width / base_height) - (target_width / target_height)))
        if best_score is None or score < best_score:
            best_score = score
            best_size = (base_width, base_height)

    return best_size or (1024, 1024)


def _image_bytes_from_result_url(url: str) -> Tuple[bytes, str]:
    if url.startswith("data:"):
        header, b64_data = url.split(",", 1)
        mime = header.split(";")[0].replace("data:", "") or "image/png"
        return base64.b64decode(b64_data), mime
    raise ValueError("Resultado externo precisa ser baixado antes do pós-processamento.")


def _result_url_from_image_bytes(image_bytes: bytes, mime: str = "image/png") -> str:
    return _data_uri_from_b64(base64.b64encode(image_bytes).decode("utf-8"), mime)




def _trim_uniform_borders(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    width, height = rgba.size
    if width < 8 or height < 8:
        return rgba

    corners = [
        rgba.getpixel((0, 0)),
        rgba.getpixel((width - 1, 0)),
        rgba.getpixel((0, height - 1)),
        rgba.getpixel((width - 1, height - 1)),
    ]

    base = corners[0]

    def _similar(a, b, tolerance: int = 16) -> bool:
        return all(abs(int(a[idx]) - int(b[idx])) <= tolerance for idx in range(4))

    if sum(1 for corner in corners if _similar(corner, base)) < 3:
        return rgba

    background = Image.new("RGBA", rgba.size, base)
    diff = ImageChops.difference(rgba, background)
    bbox = diff.getbbox()
    if not bbox:
        return rgba

    left, top, right, bottom = bbox
    trimmed_width = right - left
    trimmed_height = bottom - top

    if trimmed_width <= 0 or trimmed_height <= 0:
        return rgba

    if left == 0 and top == 0 and right == width and bottom == height:
        return rgba

    if trimmed_width < width * 0.55 or trimmed_height < height * 0.55:
        return rgba

    return rgba.crop(bbox)

def _resize_and_crop_image_bytes(image_bytes: bytes, target_width: int, target_height: int) -> bytes:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            prepared = img.convert("RGBA") if img.mode not in {"RGB", "RGBA"} else img.copy()
            prepared = _trim_uniform_borders(prepared)
            fitted = ImageOps.fit(
                prepared,
                (target_width, target_height),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
            buffer = io.BytesIO()
            fitted.save(buffer, format="PNG", optimize=True)
            return buffer.getvalue()
    except UnidentifiedImageError as exc:
        raise ValueError(f"Não foi possível interpretar a imagem retornada para pós-processamento: {exc}")


async def _apply_postprocess_if_needed(
    client: httpx.AsyncClient,
    result: Dict[str, Any],
    target_dimensions: Optional[Tuple[int, int]],
) -> Dict[str, Any]:
    if not target_dimensions:
        return result

    target_width, target_height = target_dimensions
    url = result.get("url")
    if not url:
        return result

    if url.startswith("data:"):
        source_bytes, _ = _image_bytes_from_result_url(url)
    else:
        response = await client.get(url)
        response.raise_for_status()
        source_bytes = response.content

    processed_bytes = _resize_and_crop_image_bytes(source_bytes, target_width, target_height)
    next_result = dict(result)
    next_result["url"] = _result_url_from_image_bytes(processed_bytes, "image/png")
    next_result["postprocessed"] = True
    next_result["target_dimensions"] = {"width": target_width, "height": target_height}
    next_result["motor"] = f"{result.get('motor', 'Imagem')} + Resize exato"
    return next_result


def _size_label(width: int, height: int) -> str:
    return f"{width}x{height}"

def _sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _clamp_text(text: str, max_len: int = 7000) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def _parse_json_safe(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        cleaned = text.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)


def _data_uri_from_b64(b64_data: str, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{b64_data}"


def _normalize_quality(qualidade: str) -> str:
    q = (qualidade or "").strip().lower()
    if q in {"baixa", "low", "economica", "econômica"}:
        return "low"
    if q in {"media", "média", "medium", "equilibrada"}:
        return "medium"
    return "high"


def _quality_label(q: str) -> str:
    return {"low": "Baixa", "medium": "Média", "high": "Alta"}.get(q, "Alta")


def _normalize_aspect_ratio(formato: str) -> str:
    mapping = {
        "quadrado_1_1": "1:1",
        "vertical_9_16": "9:16",
        "horizontal_16_9": "16:9",
    }
    return mapping.get((formato or "").strip(), "1:1")


def _asset_type_from_context(where: str, aspect_ratio: str) -> str:
    p = (where or "").lower()

    if "thumbnail" in p or "youtube" in p:
        return "thumbnail"
    if "story" in p or "status" in p or aspect_ratio == "9:16":
        return "story_cta"
    if "landing" in p or "site" in p or "banner" in p or aspect_ratio == "16:9":
        return "landing_banner"
    if "carrossel" in p:
        return "carousel_cover"
    if "linkedin" in p:
        return "social_media_post"

    return "feed_offer"


def _marketing_preset(asset_type: str, where: str) -> Dict[str, str]:
    destination = where or "mídia digital"

    presets = {
        "feed_offer": {
            "mode": "direct_response",
            "goal": f"peça publicitária para feed em {destination}",
            "layout": "foco visual dominante, área limpa para headline no terço superior, apoio visual central, boa distribuição de peso, sem áreas mortas e com composição equilibrada para anúncio real",
            "style": "publicidade premium, acabamento comercial de alta conversão, hierarquia visual limpa, contraste forte e leitura imediata",
            "overlay": "headline no topo, subheadline no centro ou logo abaixo, CTA opcional na base",
            "grid": "grid de anúncio social em 12 colunas, margens seguras entre 6 e 8 por cento, alinhamento limpo e ritmo visual consistente",
        },
        "story_cta": {
            "mode": "lead_generation",
            "goal": f"criativo vertical de conversão para {destination}",
            "layout": "composição vertical mobile-first, foco central ou ligeiramente abaixo do centro, headline no terço superior, subheadline no miolo, zona de CTA inferior, boa respiração visual e sem poluição",
            "style": "anúncio vertical premium, contraste forte, leitura rápida, sensação de peça profissional feita para conversão",
            "overlay": "headline no terço superior, subheadline no meio, CTA na faixa inferior",
            "grid": "grid vertical com grandes áreas seguras no topo e na base, laterais limpas e distribuição pensada para interfaces mobile",
        },
        "landing_banner": {
            "mode": "premium_branding",
            "goal": f"hero banner para {destination}",
            "layout": "layout horizontal premium, bloco de texto reservado em um lado e assunto visual no lado oposto, equilíbrio forte, espaço nobre para copy e sem vazios inúteis",
            "style": "branding comercial premium, presença corporativa forte, acabamento sofisticado e direção visual limpa",
            "overlay": "headline principal, subheadline logo abaixo, CTA opcional",
            "grid": "grid horizontal de hero com painel reservado para texto e forte separação entre copy e imagem",
        },
        "thumbnail": {
            "mode": "social_media_post",
            "goal": f"thumbnail ou capa com alto potencial de clique para {destination}",
            "layout": "foco dominante, hierarquia agressiva, fundo simplificado, headline curta e legível, contraste alto e leitura instantânea",
            "style": "acabamento de thumbnail premium, visual forte, impacto imediato e baixo ruído",
            "overlay": "headline curta com apoio opcional mínimo",
            "grid": "grid de thumbnail com foco muito claro, título forte e fundo subordinado ao elemento principal",
        },
        "carousel_cover": {
            "mode": "carousel_visual",
            "goal": f"capa de carrossel para {destination}",
            "layout": "composição editorial com grande zona de título, ponto focal memorável, boa ancoragem do visual principal e distribuição limpa do espaço",
            "style": "post editorial premium, aparência de marca forte, layout limpo e alta clareza",
            "overlay": "headline forte e apoio curto opcional",
            "grid": "grid editorial de capa com bloco de título dominante e ponto focal muito bem resolvido",
        },
        "social_media_post": {
            "mode": "social_media_post",
            "goal": f"post profissional para {destination}",
            "layout": "equilíbrio entre branding e leitura, assunto visual claro, zonas seguras para texto, boa hierarquia e sem espaço morto",
            "style": "design limpo, premium e corporativo, com acabamento polido e boa legibilidade",
            "overlay": "headline e linha de apoio curta",
            "grid": "grid social balanceado, com margens seguras e organização forte entre imagem e texto",
        },
    }

    return presets.get(asset_type, presets["feed_offer"])


def _openai_size_from_aspect_ratio(ar: str) -> str:
    if ar == "9:16":
        return "1024x1536"
    if ar == "16:9":
        return "1536x1024"
    return "1024x1024"


def _sanitize_copy(text: str, max_len: int) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text[:max_len]


def _build_user_brief(payload: ImageEngineRequest) -> str:
    parts = [
        f"Formato: {payload.formato}",
        "Objetivo de publicação: livre, sem canal fixo",
        f"Qualidade desejada: {payload.qualidade}",
        f"Paleta de cores: {payload.paleta_cores}",
    ]

    if payload.width and payload.height:
        parts.append(f"Tamanho final customizado: {payload.width}x{payload.height}")

    if payload.headline.strip():
        parts.append(f"Headline exata: {payload.headline.strip()}")

    if payload.subheadline.strip():
        parts.append(f"Sub-headline exata: {payload.subheadline.strip()}")

    if payload.descricao_visual.strip():
        parts.append(f"Descrição visual da arte: {payload.descricao_visual.strip()}")

    return "\n".join(parts)

def _build_user_edit_brief(payload: ImageEditRequest) -> str:
    parts = [
        f"Formato: {payload.formato}",
        f"Qualidade desejada: {payload.qualidade}",
    ]

    if payload.width and payload.height:
        parts.append(f"Tamanho final customizado: {payload.width}x{payload.height}")

    if payload.instrucoes_edicao.strip():
        parts.append(f"Instruções de edição: {payload.instrucoes_edicao.strip()}")

    return "\n".join(parts)

def _guess_image_content_type(filename: str, upload_content_type: Optional[str] = None) -> str:
    allowed = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if upload_content_type in allowed:
        return "image/jpeg" if upload_content_type == "image/jpg" else upload_content_type

    name = (filename or "").lower()
    if name.endswith(".png"):
        return "image/png"
    if name.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


def _validate_reference_image(image_bytes: bytes, content_type: str) -> None:
    allowed = {"image/png", "image/jpeg", "image/webp"}
    if content_type not in allowed:
        raise ValueError("Formato inválido. Use PNG, JPG, JPEG ou WEBP.")

    if not image_bytes:
        raise ValueError("A imagem de referência está vazia.")

    if len(image_bytes) > 20 * 1024 * 1024:
        raise ValueError("A imagem de referência excede 20 MB.")


async def _post_multipart_with_retry(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    data: Dict[str, Any],
    files: List[tuple],
    retries: int = 3,
    backoff_base: float = 1.2,
) -> httpx.Response:
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            resp = await client.post(url, headers=headers, data=data, files=files)
            if resp.status_code < 400:
                return resp

            if resp.status_code not in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}: {resp.text}",
                    request=resp.request,
                    response=resp,
                )

            last_exc = httpx.HTTPStatusError(
                f"HTTP {resp.status_code}: {resp.text}",
                request=resp.request,
                response=resp,
            )

        except Exception as e:
            last_exc = e

        if attempt < retries:
            await asyncio.sleep(backoff_base * attempt)

    raise last_exc if last_exc else RuntimeError("Falha desconhecida em _post_multipart_with_retry")



async def _post_json_with_retry(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    json_payload: Dict[str, Any],
    retries: int = 3,
    backoff_base: float = 1.2,
) -> httpx.Response:
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            resp = await client.post(url, headers=headers, json=json_payload)
            if resp.status_code < 400:
                return resp

            if resp.status_code not in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}: {resp.text}",
                    request=resp.request,
                    response=resp,
                )

            last_exc = httpx.HTTPStatusError(
                f"HTTP {resp.status_code}: {resp.text}",
                request=resp.request,
                response=resp,
            )

        except Exception as e:
            last_exc = e

        if attempt < retries:
            await asyncio.sleep(backoff_base * attempt)

    raise last_exc if last_exc else RuntimeError("Falha desconhecida em _post_json_with_retry")


async def _improve_prompt_with_openai(
    client: httpx.AsyncClient,
    payload: ImageEngineRequest,
    aspect_ratio: str,
    asset_type: str,
    preset: Dict[str, str],
    openai_key: str,
) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json",
    }

    headline = _sanitize_copy(payload.headline, 140)
    subheadline = _sanitize_copy(payload.subheadline, 220)
    descricao_visual = _sanitize_copy(payload.descricao_visual, 2000)

    default_copy_policy = (
        "usar_textos_exatos_do_usuario_sem_traduzir"
        if (headline or subheadline)
        else "reservar_zonas_de_texto_sem_inventar_copy"
    )

    system_text = """
Você é um diretor de arte sênior de marketing e engenheiro de prompts para geração de imagens publicitárias.

Sua função não é escrever um texto bonito.
Sua função é projetar um prompt de geração de imagem extremamente forte para publicidade real.

Retorne SOMENTE JSON válido com esta estrutura exata:
{
  "prompt_final": string,
  "negative_prompt": string,
  "creative_direction": string,
  "layout_notes": string,
  "marketing_mode": string,
  "overlay_recommendation": string,
  "design_system": string,
  "grid_spec": string,
  "text_distribution_rules": string,
  "copy_policy": string
}

Regras obrigatórias:
1. Escreva TUDO em português do Brasil.
2. O foco é marketing, conversão, direção de arte publicitária e usabilidade comercial.
3. O prompt deve dizer COMO a imagem deve ser composta, não apenas O QUE mostrar.
4. Projete uma imagem com aparência de anúncio premium, e não uma arte genérica de IA.
5. Reforce:
   - hierarquia visual forte
   - composição disciplinada
   - separação clara entre foco principal e fundo
   - iluminação comercial
   - contraste publicitário
   - nitidez premium
   - escala correta do elemento principal
   - ausência de espaços mortos
   - ausência de poluição visual
6. Se headline e sub-headline existirem, o sistema deve priorizar EXATAMENTE esses textos e NUNCA traduzi-los.
7. Nunca invente textos promocionais em inglês.
8. Nunca gere frases como Shop now, Learn more, Join now, Special offer ou equivalentes.
9. Se houver risco de tipografia ruim, preserve áreas limpas para texto ao invés de inventar textos errados.
10. O negative_prompt deve bloquear:
   - texto em inglês
   - texto aleatório
   - letras deformadas
   - tipografia ruim
   - espaços mortos
   - layout confuso
   - excesso de elementos
   - anatomia deformada
   - duplicações
   - baixa nitidez
   - visual de mockup amador
   - acabamento fraco
11. marketing_mode deve ser um destes:
   - direct_response
   - premium_branding
   - social_media_post
   - carousel_visual
   - lead_generation
12. copy_policy deve ser um destes:
   - usar_textos_exatos_do_usuario_sem_traduzir
   - reservar_zonas_de_texto_sem_inventar_copy
   - usar_copy_curta_em_portugues
13. text_distribution_rules deve mencionar:
   - máximo de linhas da headline
   - máximo de linhas da sub-headline
   - margens seguras
   - equilíbrio texto versus imagem
   - proibição de blocos longos
14. O resultado precisa ser mais forte, mais técnico e mais publicitário do que um prompt comum.
"""

    user_text = (
        f"Briefing estruturado:\n{_build_user_brief(payload)}\n\n"
        f"Aspect ratio: {aspect_ratio}\n"
        f"Tipo de peça: {asset_type}\n"
        f"Objetivo do preset: {preset['goal']}\n"
        f"Comportamento de layout do preset: {preset['layout']}\n"
        f"Estilo visual do preset: {preset['style']}\n"
        f"Recomendação de overlay do preset: {preset['overlay']}\n"
        f"Grid base do preset: {preset['grid']}\n\n"
        f"Headline exata do usuário: {headline or 'não informada'}\n"
        f"Sub-headline exata do usuário: {subheadline or 'não informada'}\n"
        f"Descrição visual: {descricao_visual or 'não informada'}\n\n"
        "Quero um refinamento com foco em direção de arte publicitária, acabamento premium, composição forte, legibilidade real, linguagem em português do Brasil e proibição total de textos falsos em inglês."
    )

    payload_json = {
        "model": OPENAI_CHAT_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.15,
    }

    resp = await _post_json_with_retry(
        client=client,
        url="https://api.openai.com/v1/chat/completions",
        headers=headers,
        json_payload=payload_json,
        retries=3,
    )

    content = resp.json()["choices"][0]["message"]["content"]
    data = _parse_json_safe(content)

    return {
        "prompt_final": _clamp_text(data.get("prompt_final", "")),
        "negative_prompt": _clamp_text(data.get("negative_prompt", "")),
        "creative_direction": _clamp_text(data.get("creative_direction", "")),
        "layout_notes": _clamp_text(data.get("layout_notes", "")),
        "marketing_mode": _clamp_text(data.get("marketing_mode", preset["mode"])),
        "overlay_recommendation": _clamp_text(data.get("overlay_recommendation", "")),
        "design_system": _clamp_text(data.get("design_system", "")),
        "grid_spec": _clamp_text(data.get("grid_spec", "")),
        "text_distribution_rules": _clamp_text(
            data.get(
                "text_distribution_rules",
                "headline com no máximo 2 linhas, sub-headline com no máximo 3 linhas, margens seguras, sem blocos longos e com proporção equilibrada entre texto e imagem",
            )
        ),
        "copy_policy": _clamp_text(data.get("copy_policy", default_copy_policy)),
    }


def _build_final_generation_prompt(
    payload: ImageEngineRequest,
    aspect_ratio: str,
    asset_type: str,
    preset: Dict[str, str],
    improved: Dict[str, str],
) -> str:
    headline = _sanitize_copy(payload.headline, 140)
    subheadline = _sanitize_copy(payload.subheadline, 220)
    description = _sanitize_copy(payload.descricao_visual, 3000)

    copy_block = []
    if headline:
        copy_block.append(f"- Use exatamente esta headline em português do Brasil, sem traduzir: {headline}")
    if subheadline:
        copy_block.append(f"- Use exatamente esta sub-headline em português do Brasil, sem traduzir: {subheadline}")
    if not headline and not subheadline:
        copy_block.append("- Não há headline ou sub-headline fixas. Preserve áreas nobres para texto, mas não invente parágrafos ou slogans falsos.")

    copy_block.append("- Nunca substituir os textos do usuário por versões em inglês.")
    copy_block.append("- Nunca inserir frases como Shop now, Learn more, Join now, New collection ou qualquer placeholder em inglês.")
    copy_block.append("- Se a engine não conseguir renderizar o texto com qualidade, priorize composição limpa e zonas reservadas, em vez de inventar texto ruim.")

    final_prompt = f"""
{improved['prompt_final']}

Esta imagem deve parecer uma peça publicitária premium, real e profissional.
Não criar uma arte genérica de IA.
Não criar um mockup fraco.
Não criar uma imagem bonita porém inútil para marketing.

Objetivo principal:
- gerar uma imagem com cara de anúncio de alta conversão
- forte impacto visual
- composição disciplinada
- leitura imediata
- acabamento premium
- hierarquia clara
- alto valor percebido

Contexto estruturado da peça:
- formato selecionado: {payload.formato}
- proporção final: {aspect_ratio}
- tipo de peça: {asset_type}
- destino de publicação: livre, sem canal fixo
- nível de qualidade solicitado: {_quality_label(_normalize_quality(payload.qualidade))}
- paleta de cores: {payload.paleta_cores}
- descrição visual solicitada: {description or 'seguir uma direção comercial premium coerente com o briefing'}
- tamanho final desejado: {_size_label(payload.width, payload.height) if payload.width and payload.height else 'usar o canvas padrão do formato selecionado'}

Objetivo do preset:
{preset['goal']}

Comportamento obrigatório de layout:
{preset['layout']}

Estilo visual obrigatório:
{preset['style']}

Direção criativa:
{improved['creative_direction']}

Notas de layout:
{improved['layout_notes']}

Recomendação de overlay:
{improved['overlay_recommendation']}

Sistema visual:
{improved['design_system']}

Especificação de grid:
{improved['grid_spec']}

Regras de distribuição de texto:
{improved['text_distribution_rules']}

Política de copy:
{improved['copy_policy']}

Regras técnicas de composição:
- compor como diretor de arte publicitário, não como artista aleatório
- criar um ponto focal dominante e imediatamente compreensível
- controlar escala do elemento principal para que ele tenha presença forte
- separar bem objeto principal e fundo
- usar iluminação comercial e acabamento premium
- trabalhar profundidade de cena de forma elegante, sem poluir a leitura
- manter contraste suficiente para uma headline clara
- manter a área da sub-headline mais calma do que o centro focal
- evitar fundo excessivamente carregado atrás do texto
- evitar espaços mortos, vazios acidentais ou cantos sem função
- evitar excesso de mini elementos concorrendo com o foco
- criar equilíbrio entre sofisticação visual e clareza de marketing
- preservar margens seguras para recortes da plataforma
- manter sensação de peça pronta para campanha real
- nitidez premium, materiais bem resolvidos, reflexos controlados, contraste publicitário, acabamento comercial de alto padrão

Regras obrigatórias de idioma e texto:
{chr(10).join(copy_block)}

Restrições fortes:
- todo texto visível deve estar em português do Brasil
- não traduzir os textos do usuário
- não resumir o texto do usuário
- não inventar slogans em inglês
- não encher a peça com labels falsos
- não usar parágrafos longos
- não destruir a composição tentando encaixar texto demais
"""
    return _clamp_text(final_prompt, 7000)


async def _generate_openai_image(
    client: httpx.AsyncClient,
    final_prompt: str,
    aspect_ratio: str,
    quality: str,
    openai_key: str,
    openai_size: Optional[str] = None,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENAI_IMAGE_MODEL,
        "prompt": final_prompt,
        "size": openai_size or _openai_size_from_aspect_ratio(aspect_ratio),
        "quality": quality,
        "output_format": "png",
        "background": "opaque",
    }

    resp = await _post_json_with_retry(
        client=client,
        url="https://api.openai.com/v1/images/generations",
        headers=headers,
        json_payload=payload,
        retries=3,
    )

    body = resp.json()
    data = body.get("data", [])
    if not data:
        raise ValueError(f"OpenAI sem data: {body}")

    first = data[0]
    b64_json = first.get("b64_json")
    if not b64_json:
        raise ValueError(f"OpenAI não retornou b64_json: {body}")

    return {
        "engine_id": "openai",
        "motor": "OpenAI GPT Image 1.5",
        "url": _data_uri_from_b64(b64_json, "image/png"),
        "raw": body,
    }


async def _generate_flux_image(
    client: httpx.AsyncClient,
    final_prompt: str,
    negative_prompt: str,
    aspect_ratio: str,
    fal_key: str,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Key {fal_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "prompt": final_prompt,
        "negative_prompt": negative_prompt or None,
        "aspect_ratio": aspect_ratio,
        "num_images": 1,
        "output_format": "jpeg",
        "safety_tolerance": 2,
    }

    resp = await _post_json_with_retry(
        client=client,
        url=f"https://fal.run/{FAL_MODEL_PATH}",
        headers=headers,
        json_payload=payload,
        retries=3,
    )

    body = resp.json()
    images = body.get("images", [])
    if not images or not images[0].get("url"):
        raise ValueError(f"FLUX não retornou URL válida: {body}")

    return {
        "engine_id": "flux",
        "motor": "FLUX 1.1 Pro Ultra",
        "url": images[0]["url"],
        "raw": body,
    }


def _extract_gemini_inline_image(response_json: Dict[str, Any]) -> Optional[str]:
    candidates = response_json.get("candidates", [])
    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        for part in parts:
            inline_data = part.get("inlineData") or part.get("inline_data")
            if inline_data and inline_data.get("data"):
                mime = inline_data.get("mimeType") or inline_data.get("mime_type") or "image/png"
                return _data_uri_from_b64(inline_data["data"], mime)
    return None


async def _generate_google_native_image(
    client: httpx.AsyncClient,
    final_prompt: str,
    aspect_ratio: str,
    gemini_key: str,
    model_name: str,
) -> Dict[str, Any]:
    headers = {
        "x-goog-api-key": gemini_key,
        "Content-Type": "application/json",
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": final_prompt}]}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": "2K",
            },
        },
    }

    resp = await _post_json_with_retry(client, url, headers, payload, retries=3)
    body = resp.json()
    data_uri = _extract_gemini_inline_image(body)
    if not data_uri:
        raise ValueError(f"{model_name} não retornou inline image válida: {body}")

    pretty_name = "Google Nano Banana Pro" if model_name == GEMINI_NATIVE_PRO_MODEL else "Google Nano Banana 2"

    return {
        "engine_id": "google",
        "motor": pretty_name,
        "google_model": model_name,
        "url": data_uri,
        "raw": body,
    }


async def _generate_google_imagen_image(
    client: httpx.AsyncClient,
    final_prompt: str,
    aspect_ratio: str,
    gemini_key: str,
) -> Dict[str, Any]:
    headers = {
        "x-goog-api-key": gemini_key,
        "Content-Type": "application/json",
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GOOGLE_IMAGEN_MODEL}:predict"

    payload = {
        "instances": [{"prompt": final_prompt}],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": aspect_ratio,
            "imageSize": "2K",
            "personGeneration": "allow_adult",
        },
    }

    resp = await _post_json_with_retry(client, url, headers, payload, retries=3)
    body = resp.json()

    predictions = body.get("predictions", [])
    if not predictions:
        raise ValueError(f"Google Imagen sem predictions: {body}")

    pred = predictions[0]
    base64_img = pred.get("bytesBase64Encoded")
    if not base64_img:
        raise ValueError(f"Google Imagen sem bytesBase64Encoded: {body}")

    return {
        "engine_id": "google",
        "motor": "Google Imagen 4 Ultra",
        "google_model": GOOGLE_IMAGEN_MODEL,
        "url": _data_uri_from_b64(base64_img, "image/png"),
        "raw": body,
    }


async def _generate_google_best_available(
    client: httpx.AsyncClient,
    final_prompt: str,
    aspect_ratio: str,
    gemini_key: str,
) -> Dict[str, Any]:
    errors = []

    try:
        return await _generate_google_native_image(
            client,
            final_prompt,
            aspect_ratio,
            gemini_key,
            GEMINI_NATIVE_PRO_MODEL,
        )
    except Exception as e:
        errors.append(f"{GEMINI_NATIVE_PRO_MODEL}: {str(e)}")

    try:
        return await _generate_google_native_image(
            client,
            final_prompt,
            aspect_ratio,
            gemini_key,
            GEMINI_NATIVE_FAST_MODEL,
        )
    except Exception as e:
        errors.append(f"{GEMINI_NATIVE_FAST_MODEL}: {str(e)}")

    try:
        return await _generate_google_imagen_image(
            client,
            final_prompt,
            aspect_ratio,
            gemini_key,
        )
    except Exception as e:
        errors.append(f"{GOOGLE_IMAGEN_MODEL}: {str(e)}")

    raise ValueError(" | ".join(errors))



async def _improve_edit_prompt_with_openai(
    client: httpx.AsyncClient,
    payload: ImageEditRequest,
    aspect_ratio: str,
    openai_key: str,
) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json",
    }

    instrucoes_edicao = _sanitize_copy(payload.instrucoes_edicao, 2600)

    system_text = """
Você é um diretor de arte sênior de pós-produção e engenheiro de prompts para edição de imagens com referência.

Sua função é escrever um prompt de edição extremamente preciso para uma imagem já existente.
A prioridade não é reinventar a peça. A prioridade é preservar o original e modificar somente o que foi pedido.

Retorne SOMENTE JSON válido com esta estrutura exata:
{
  "prompt_final": string,
  "negative_prompt": string,
  "creative_direction": string,
  "layout_notes": string,
  "preservation_rules": string,
  "edit_strategy": string,
  "micro_detail_rules": string,
  "consistency_rules": string
}

Regras obrigatórias:
1. Escreva TUDO em português do Brasil.
2. Trate a imagem enviada como base dominante e autoritativa.
3. Preservar tudo o que não foi explicitamente pedido para mudar.
4. A edição deve ser local e precisa, evitando reconstrução total da peça.
5. Se houver logos, selos, ícones, marcas, tipografias pequenas, estampas, embalagens, assinaturas visuais ou detalhes delicados, priorize preservar exatamente o que já existe em vez de redesenhar, reinterpretar ou recriar.
6. Não substituir logos pequenos por versões novas, aproximadas ou genéricas.
7. Não inventar elementos de branding, não redesenhar marcas e não trocar símbolos existentes por versões parecidas.
8. Se algum detalhe pequeno não precisar mudar, ele deve permanecer visualmente consistente com a referência original.
9. O negative_prompt deve bloquear: recriação completa da cena, redesenho de logos, troca de marca, texto aleatório, texto em inglês, duplicações, deformações, mudanças arbitrárias no produto, mudanças desnecessárias no enquadramento, alterações indevidas de cor da marca, remoção de detalhes importantes, blur, baixa nitidez e aparência genérica de IA.
10. preservation_rules deve reforçar a preservação de identidade visual, branding, embalagem, produto, personagem, enquadramento, perspectiva, materiais e microdetalhes sempre que isso não conflitar com o pedido.
11. edit_strategy deve descrever edição pontual, incremental e controlada, nunca recriação ampla sem necessidade.
12. micro_detail_rules deve explicar como proteger logos, textos pequenos, selos, ícones, botões, acabamentos, costuras, rótulos e elementos gráficos finos.
13. consistency_rules deve explicar como manter coerência entre fundo, foco principal, sombras, reflexos, perspectiva e proporções.
14. O resultado precisa ser mais técnico, mais restritivo e mais útil para edição real do que um prompt comum.
"""

    user_text = (
        f"Briefing estruturado para edição:\n{_build_user_edit_brief(payload)}\n\n"
        f"Aspect ratio de saída: {aspect_ratio}\n\n"
        f"Instruções do usuário: {instrucoes_edicao or 'não informadas'}\n\n"
        "Quero um refinamento com foco em edição fiel, intervenção precisa, preservação de branding e proteção máxima de logos pequenos e detalhes sensíveis."
    )

    payload_json = {
        "model": OPENAI_CHAT_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.1,
    }

    resp = await _post_json_with_retry(
        client=client,
        url="https://api.openai.com/v1/chat/completions",
        headers=headers,
        json_payload=payload_json,
        retries=3,
    )

    content = resp.json()["choices"][0]["message"]["content"]
    data = _parse_json_safe(content)

    return {
        "prompt_final": _clamp_text(data.get("prompt_final", "")),
        "negative_prompt": _clamp_text(data.get("negative_prompt", "")),
        "creative_direction": _clamp_text(data.get("creative_direction", "edição precisa com preservação máxima da base original")),
        "layout_notes": _clamp_text(data.get("layout_notes", "preservar enquadramento e composição geral, mudando somente o que foi solicitado")),
        "preservation_rules": _clamp_text(data.get("preservation_rules", "preservar identidade visual, branding, embalagem, detalhes finos, logos e materiais originais sempre que não houver pedido explícito de alteração")),
        "edit_strategy": _clamp_text(data.get("edit_strategy", "aplicar edição localizada, incremental e controlada, sem recriar a peça inteira")),
        "micro_detail_rules": _clamp_text(data.get("micro_detail_rules", "não redesenhar logos pequenos, selos, ícones, rótulos ou detalhes gráficos finos; reaproveitar visualmente o que já existe na referência sempre que possível")),
        "consistency_rules": _clamp_text(data.get("consistency_rules", "manter coerência de perspectiva, escala, luz, sombras, reflexos, nitidez e cor entre os elementos preservados e os editados")),
    }

def _build_final_edit_prompt(
    payload: ImageEditRequest,
    aspect_ratio: str,
    improved: Dict[str, str],
) -> str:
    instructions = _sanitize_copy(payload.instrucoes_edicao, 3400)

    final_prompt = f"""
{improved['prompt_final']}

Esta tarefa é uma EDIÇÃO de imagem baseada em referência.
Use a imagem enviada como base principal e dominante da composição.
Preserve tudo o que não foi explicitamente pedido para mudar.
Não recriar a peça inteira.
Não reinterpretar a marca.
Não redesenhar detalhes pequenos sem necessidade.

Objetivo principal:
- editar a imagem com precisão
- preservar o máximo possível do original
- alterar apenas os pontos necessários para cumprir o briefing
- manter acabamento premium e coerência visual
- evitar qualquer reconstrução desnecessária de logos, marcas, selos, ícones, embalagens e detalhes sensíveis

Contexto estruturado da edição:
- formato selecionado: {payload.formato}
- proporção final: {aspect_ratio}
- nível de qualidade solicitado: {_quality_label(_normalize_quality(payload.qualidade))}
- instruções de edição: {instructions or 'seguir a imagem de referência e editar somente o necessário'}
- tamanho final desejado: {_size_label(payload.width, payload.height) if payload.width and payload.height else 'usar o canvas padrão do formato selecionado'}

Direção criativa:
{improved['creative_direction']}

Notas de layout:
{improved['layout_notes']}

Regras de preservação:
{improved['preservation_rules']}

Estratégia de edição:
{improved['edit_strategy']}

Proteção de microdetalhes:
{improved['micro_detail_rules']}

Regras de consistência:
{improved['consistency_rules']}

Regras técnicas obrigatórias:
- tratar a imagem enviada como referência dominante
- modificar somente o que foi solicitado nas instruções
- manter enquadramento, perspectiva e proporção sempre que possível
- preservar identidade visual, produto, embalagem, materiais e estrutura original
- preservar exatamente logos, marcas, selos, ícones, assinaturas visuais e detalhes pequenos que não precisem ser alterados
- não substituir logos pequenos por versões novas, aproximadas, borradas ou genéricas
- não inventar branding novo
- não simplificar elementos pequenos importantes
- não apagar detalhes finos relevantes
- se houver intervenção próxima a uma logo ou detalhe delicado, manter forma, posição relativa, nitidez e leitura consistentes com a referência
- manter sombras, reflexos, contraste, textura e iluminação coerentes com a base original
- evitar deformações, duplicações, desalinhamentos, artefatos e aparência genérica de IA
- o resultado deve parecer a mesma peça refinada, e não outra peça recriada do zero

Restrições fortes:
- não recriar a cena inteira sem necessidade
- não redesenhar ou trocar logos
- não trocar marca, símbolo, selo ou rótulo existente por algo parecido
- não mudar cores de branding sem pedido explícito
- não adicionar texto aleatório
- não inserir texto em inglês
- não poluir o layout
"""
    return _clamp_text(final_prompt, 7000)


def _build_localized_prompt_appendix(
    localized_analysis: Optional[Dict[str, Any]],
    instruction_info: Optional[Dict[str, Any]] = None,
) -> str:
    if not localized_analysis:
        return ""

    lines = []
    target = localized_analysis.get("target_text") or ""
    replacement = localized_analysis.get("replacement_text") or ""
    operation = localized_analysis.get("operation") or "text_replace"
    confidence = float(localized_analysis.get("confidence", 0.0) or 0.0)
    bbox = localized_analysis.get("bbox")
    text_bbox = localized_analysis.get("text_bbox")
    container_bbox = localized_analysis.get("container_bbox")

    lines.append("\n\n--- INSTRUÇÃO LOCALIZADA DE EDIÇÃO ---")

    if target and replacement:
        lines.append(f"Substitua o texto \"{target}\" por \"{replacement}\".")

    if operation == "button_text_replace":
        lines.append("Este texto está dentro de um botão, badge ou chip. Preserve a forma, cor e bordas do elemento container.")

    region = text_bbox or bbox or container_bbox
    if region:
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)
        lines.append(
            f"Região alvo (coordenadas normalizadas): x={x:.4f}, y={y:.4f}, largura={w:.4f}, altura={h:.4f}. "
            "Edite somente essa região. Preserve absolutamente tudo o que está fora dela."
        )

    if container_bbox and operation == "button_text_replace":
        cx = container_bbox.get("x", 0)
        cy = container_bbox.get("y", 0)
        cw = container_bbox.get("w", 0)
        ch = container_bbox.get("h", 0)
        lines.append(
            f"Container do botão (coordenadas normalizadas): x={cx:.4f}, y={cy:.4f}, largura={cw:.4f}, altura={ch:.4f}. "
            "Recrie o botão com o novo texto mantendo exatamente a mesma aparência visual."
        )

    style = localized_analysis.get("style") or {}
    style_notes = []
    if style.get("text_color"):
        style_notes.append(f"cor do texto: {style['text_color']}")
    if style.get("background_color"):
        style_notes.append(f"cor de fundo: {style['background_color']}")
    if style.get("font_weight"):
        style_notes.append(f"peso da fonte: {style['font_weight']}")
    if style.get("alignment"):
        style_notes.append(f"alinhamento: {style['alignment']}")
    if style.get("shadow"):
        style_notes.append("com sombra")
    if style.get("glow"):
        style_notes.append("com brilho/glow")
    if style_notes:
        lines.append("Estilo visual a manter: " + ", ".join(style_notes) + ".")

    if confidence >= 0.75:
        lines.append("Nível de confiança na localização: alto. Aplique a edição com precisão cirúrgica.")
    elif confidence >= 0.55:
        lines.append("Nível de confiança na localização: médio. Aplique a edição com cuidado redobrado na preservação do entorno.")
    else:
        lines.append("Nível de confiança na localização: baixo. Seja conservador — edite apenas o mínimo necessário e preserve tudo ao redor.")

    lines.append("--- FIM DA INSTRUÇÃO LOCALIZADA ---")
    return "\n".join(lines)


def _build_edit_attempt_plan(
    instruction_info: Dict[str, Any],
    localized_analysis: Optional[Dict[str, Any]],
    localized_mode: bool,
) -> Dict[str, Any]:
    use_local_render_first = should_use_local_text_render(localized_analysis, instruction_info)
    return {
        "use_local_render_first": use_local_render_first,
        "call_openai_edit": not use_local_render_first,
        "reason": "text_replace_deterministic" if use_local_render_first else ("masked_openai_edit" if localized_mode else "full_openai_edit"),
    }


async def _edit_openai_image(
    client: httpx.AsyncClient,
    image_bytes: bytes,
    filename: str,
    content_type: str,
    final_prompt: str,
    aspect_ratio: str,
    quality: str,
    openai_key: str,
    openai_size: Optional[str] = None,
    mask_bytes: Optional[bytes] = None,
    input_fidelity: str = "high",
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {openai_key}",
    }

    data = {
        "model": OPENAI_IMAGE_MODEL,
        "prompt": final_prompt,
        "size": openai_size or _openai_size_from_aspect_ratio(aspect_ratio),
        "quality": quality,
        "output_format": "png",
        "input_fidelity": input_fidelity,
        "background": "opaque",
    }

    files = [
        ("image[]", (filename or "reference.png", image_bytes, content_type)),
    ]
    if mask_bytes:
        files.append(("mask", ("localized-mask.png", mask_bytes, "image/png")))

    resp = await _post_multipart_with_retry(
        client=client,
        url="https://api.openai.com/v1/images/edits",
        headers=headers,
        data=data,
        files=files,
        retries=3,
    )

    body = resp.json()
    data_items = body.get("data", [])
    if not data_items:
        raise ValueError(f"OpenAI edit sem data: {body}")

    first = data_items[0]
    b64_json = first.get("b64_json")
    if not b64_json:
        raise ValueError(f"OpenAI edit não retornou b64_json: {body}")

    return {
        "engine_id": "openai_edit",
        "motor": "OpenAI GPT Image 1.5 Edit",
        "url": _data_uri_from_b64(b64_json, "image/png"),
        "raw": body,
    }
@router.post("/api/image-engine/stream")
async def image_engine_stream(body: ImageEngineRequest, current_user: User = Depends(get_current_user)):
    openai_key = os.getenv("OPENAI_API_KEY")
    fal_key = os.getenv("FAL_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    async def event_generator():
        if not openai_key:
            yield _sse({"error": "OPENAI_API_KEY não configurada."})
            return

        if not fal_key:
            yield _sse({"error": "FAL_KEY não configurada."})
            return

        if not gemini_key:
            yield _sse({"error": "GEMINI_API_KEY não configurada."})
            return

        try:
            requested_dimensions = _resolve_target_dimensions(body.width, body.height)
            aspect_ratio = _normalize_aspect_ratio(body.formato)
            openai_quality = _normalize_quality(body.qualidade)

            if requested_dimensions:
                base_width, base_height = _choose_best_supported_base_size(*requested_dimensions)
                engine_aspect_ratio = _base_size_to_aspect_ratio(base_width, base_height)
                openai_size = f"{base_width}x{base_height}"
            else:
                engine_aspect_ratio = aspect_ratio
                openai_size = _openai_size_from_aspect_ratio(engine_aspect_ratio)

            asset_type = _asset_type_from_context(None, engine_aspect_ratio)
            preset = _marketing_preset(asset_type, None)

            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
                yield _sse({
                    "status": "Analisando briefing e refinando o prompt com foco em direção de arte publicitária...",
                    "progress": 12,
                    "meta": {
                        "aspect_ratio": engine_aspect_ratio,
                        "quality": openai_quality,
                        "asset_type": asset_type,
                        "preset_mode": preset.get("mode"),
                        "openai_size": openai_size,
                        "target_dimensions": {"width": requested_dimensions[0], "height": requested_dimensions[1]} if requested_dimensions else None,
                    },
                })

                improved = await _improve_prompt_with_openai(
                    client=client,
                    payload=body,
                    aspect_ratio=engine_aspect_ratio,
                    asset_type=asset_type,
                    preset=preset,
                    openai_key=openai_key,
                )

                final_prompt = _build_final_generation_prompt(
                    payload=body,
                    aspect_ratio=engine_aspect_ratio,
                    asset_type=asset_type,
                    preset=preset,
                    improved=improved,
                )

                yield _sse({
                    "status": "Prompt refinado. Gerando nas 3 engines, sem ranking extra, com foco em qualidade visual e texto em português.",
                    "progress": 28,
                    "improved_prompt": improved["prompt_final"],
                    "negative_prompt": improved["negative_prompt"],
                    "creative_direction": improved["creative_direction"],
                    "layout_notes": improved["layout_notes"],
                    "overlay_recommendation": improved["overlay_recommendation"],
                    "grid_spec": improved["grid_spec"],
                    "final_prompt": final_prompt,
                    "aspect_ratio": engine_aspect_ratio,
                    "quality": openai_quality,
                    "asset_type": asset_type,
                    "preset_mode": preset.get("mode"),
                    "openai_size": openai_size,
                    "target_dimensions": {"width": requested_dimensions[0], "height": requested_dimensions[1]} if requested_dimensions else None,
                })

                tasks = [
                    asyncio.create_task(
                        _generate_openai_image(
                            client,
                            final_prompt,
                            engine_aspect_ratio,
                            openai_quality,
                            openai_key,
                            openai_size,
                        )
                    ),
                    asyncio.create_task(
                        _generate_flux_image(
                            client,
                            final_prompt,
                            improved["negative_prompt"],
                            engine_aspect_ratio,
                            fal_key,
                        )
                    ),
                    asyncio.create_task(
                        _generate_google_best_available(
                            client,
                            final_prompt,
                            engine_aspect_ratio,
                            gemini_key,
                        )
                    ),
                ]

                completed_results: List[Dict[str, Any]] = []
                engine_errors: List[Dict[str, Any]] = []
                total = len(tasks)
                done_count = 0

                for coro in asyncio.as_completed(tasks):
                    try:
                        result = await coro
                        result = await _apply_postprocess_if_needed(client, result, requested_dimensions)
                        completed_results.append(result)
                        done_count += 1

                        yield _sse({
                            "status": f"Imagem gerada com sucesso em {result['motor']}.",
                            "progress": 28 + int((done_count / total) * 62),
                            "partial_result": {
                                "engine_id": result["engine_id"],
                                "motor": result["motor"],
                                "url": result["url"],
                            },
                            "completed": done_count,
                            "total": total,
                        })

                    except Exception as e:
                        done_count += 1
                        engine_errors.append({"erro": str(e)})

                        yield _sse({
                            "status": "Uma das engines falhou, mas o processo continua.",
                            "progress": 28 + int((done_count / total) * 62),
                            "warning": str(e),
                            "completed": done_count,
                            "total": total,
                        })

                valid_images = [r for r in completed_results if r.get("url")]
                if not valid_images:
                    raise RuntimeError("Nenhuma engine conseguiu gerar imagem válida.")

                yield _sse({
                    "status": "Concluído. Entregando as imagens geradas.",
                    "progress": 100,
                    "improved_prompt": improved["prompt_final"],
                    "negative_prompt": improved["negative_prompt"],
                    "creative_direction": improved["creative_direction"],
                    "layout_notes": improved["layout_notes"],
                    "overlay_recommendation": improved["overlay_recommendation"],
                    "grid_spec": improved["grid_spec"],
                    "final_prompt": final_prompt,
                    "final_results": [
                        {
                            "engine_id": item["engine_id"],
                            "motor": item["motor"],
                            "url": item["url"],
                        }
                        for item in valid_images
                    ],
                    "engine_errors": engine_errors,
                })

        except Exception as e:
            yield _sse({"error": f"Erro interno no motor: {str(e)}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/api/image-engine/edit/stream")
async def image_engine_edit_stream(
    reference_image: UploadFile = File(...),
    formato: str = Form(...),
    qualidade: str = Form(...),
    instrucoes_edicao: str = Form(...),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    current_user: User = Depends(get_current_user),
):
    del current_user

    openai_key = os.getenv("OPENAI_API_KEY")

    image_bytes = await reference_image.read()
    image_filename = reference_image.filename or "reference.png"
    image_content_type = _guess_image_content_type(image_filename, reference_image.content_type)

    body = ImageEditRequest(
        formato=formato,
        qualidade=qualidade,
        instrucoes_edicao=instrucoes_edicao,
        width=width,
        height=height,
    )

    async def event_generator():
        if not openai_key:
            yield _sse({"error": "OPENAI_API_KEY não configurada."})
            return

        try:
            _validate_reference_image(image_bytes, image_content_type)

            if not body.instrucoes_edicao.strip():
                raise ValueError("As instruções de edição são obrigatórias.")

            requested_dimensions = _resolve_target_dimensions(body.width, body.height)
            aspect_ratio = _normalize_aspect_ratio(body.formato)
            openai_quality = _normalize_quality(body.qualidade)

            if requested_dimensions:
                base_width, base_height = _choose_best_supported_base_size(*requested_dimensions)
                engine_aspect_ratio = _base_size_to_aspect_ratio(base_width, base_height)
                openai_size = f"{base_width}x{base_height}"
            else:
                engine_aspect_ratio = aspect_ratio
                openai_size = _openai_size_from_aspect_ratio(engine_aspect_ratio)

            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
                yield _sse({
                    "status": "Analisando a imagem de referência e refinando o prompt de edição...",
                    "progress": 14,
                    "meta": {
                        "aspect_ratio": engine_aspect_ratio,
                        "quality": openai_quality,
                        "reference_filename": image_filename,
                        "openai_size": openai_size,
                        "target_dimensions": {"width": requested_dimensions[0], "height": requested_dimensions[1]} if requested_dimensions else None,
                    },
                })

                improved = await _improve_edit_prompt_with_openai(
                    client=client,
                    payload=body,
                    aspect_ratio=engine_aspect_ratio,
                    openai_key=openai_key,
                )

                final_prompt = _build_final_edit_prompt(
                    payload=body,
                    aspect_ratio=engine_aspect_ratio,
                    improved=improved,
                )

                yield _sse({
                    "status": "Prompt refinado. Tentando localizar a área exata da edição antes de enviar para a engine final.",
                    "progress": 46,
                    "improved_prompt": improved["prompt_final"],
                    "negative_prompt": improved["negative_prompt"],
                    "creative_direction": improved["creative_direction"],
                    "layout_notes": improved["layout_notes"],
                    "preservation_rules": improved["preservation_rules"],
                    "edit_strategy": improved["edit_strategy"],
                    "micro_detail_rules": improved["micro_detail_rules"],
                    "consistency_rules": improved["consistency_rules"],
                    "final_prompt": final_prompt,
                    "aspect_ratio": engine_aspect_ratio,
                    "quality": openai_quality,
                    "openai_size": openai_size,
                    "target_dimensions": {"width": requested_dimensions[0], "height": requested_dimensions[1]} if requested_dimensions else None,
                })

                instruction_info = extract_edit_instruction_info(body.instrucoes_edicao)
                is_multi_replace = instruction_info.get("is_multi_replace", False)
                is_pure_text_edit = instruction_info.get("is_pure_text_edit", False)

                localized_analysis = None
                all_localized_analyses: List[Dict[str, Any]] = []
                localized_mask = None
                localized_mode = False
                localized_warning = None

                try:
                    if is_multi_replace and is_pure_text_edit:
                        # Multiple text swaps: detect each region in parallel
                        all_localized_analyses = await analyze_all_regions_with_openai(
                            client=client,
                            image_bytes=image_bytes,
                            content_type=image_content_type,
                            instruction_info=instruction_info,
                            model=OPENAI_CHAT_MODEL,
                            api_key=openai_key,
                        )
                        # Use the first valid analysis as the primary for mask/compat
                        localized_analysis = all_localized_analyses[0] if all_localized_analyses else None
                        localized_mode = False  # multi-replace uses sequential local render, not masked OpenAI
                    else:
                        localized_analysis = await analyze_region_with_openai(
                            client=client,
                            image_bytes=image_bytes,
                            content_type=image_content_type,
                            instruction=body.instrucoes_edicao,
                            model=OPENAI_CHAT_MODEL,
                            api_key=openai_key,
                        )
                        all_localized_analyses = [localized_analysis] if localized_analysis else []
                        if should_use_localized_edit(localized_analysis):
                            localized_mask = build_mask_from_analysis(image_bytes, localized_analysis)
                            localized_mode = localized_mask is not None
                except Exception as region_exc:
                    localized_warning = f"Falha na detecção localizada. Seguindo com edição conservadora. Detalhe: {str(region_exc)}"

                # For multi-replace, always use local render if we have all analyses
                all_localizable = (
                    is_multi_replace
                    and is_pure_text_edit
                    and len(all_localized_analyses) == len(instruction_info.get("all_replacements", []))
                    and all(should_use_local_text_render(a, {"is_pure_text_edit": True}) for a in all_localized_analyses)
                )

                attempt_plan = _build_edit_attempt_plan(
                    instruction_info=instruction_info,
                    localized_analysis=localized_analysis,
                    localized_mode=localized_mode,
                ) if not all_localizable else {
                    "use_local_render_first": True,
                    "call_openai_edit": False,
                    "reason": "multi_text_replace_deterministic",
                }

                if attempt_plan["use_local_render_first"]:
                    n = len(all_localized_analyses)
                    yield _sse({
                        "status": f"{'Múltiplas substituições' if is_multi_replace else 'Texto'} identificado{'s' if is_multi_replace else ''} com boa confiança. Aplicando {n} edição{'ões' if n > 1 else ''} local{'is' if n > 1 else ''} determinística{'s' if n > 1 else ''} preservando 100% do layout.",
                        "progress": 62,
                        "localized_analysis": localized_analysis,
                        "localized_mode": localized_mode,
                        "warning": localized_warning,
                        "attempt_plan": attempt_plan,
                    })
                elif localized_mode:
                    yield _sse({
                        "status": "Área localizada com sucesso. Aplicando edição mascarada para preservar o restante da imagem.",
                        "progress": 62,
                        "localized_analysis": localized_analysis,
                        "localized_mode": True,
                        "warning": localized_warning,
                        "attempt_plan": attempt_plan,
                    })
                else:
                    yield _sse({
                        "status": "Não foi possível garantir uma área segura com máscara. Aplicando edição global mais conservadora, mantendo a peça original como referência dominante.",
                        "progress": 62,
                        "localized_analysis": localized_analysis,
                        "localized_mode": False,
                        "warning": localized_warning,
                        "attempt_plan": attempt_plan,
                    })

                result = None

                # --- Multi-replace: apply all replacements sequentially ---
                if all_localizable:
                    local_bytes = render_all_local_text_replacements(
                        image_bytes=image_bytes,
                        analyses=all_localized_analyses,
                    )
                    if local_bytes:
                        result = {
                            "engine_id": "local_structured_edit",
                            "motor": "Edição Local Estruturada (múltipla)",
                            "url": _data_uri_from_b64(base64.b64encode(local_bytes).decode("utf-8"), "image/png"),
                            "raw": {"strategy": attempt_plan["reason"], "replacements_count": len(all_localized_analyses)},
                        }

                # --- Single-replace local render ---
                if result is None and attempt_plan["use_local_render_first"] and localized_analysis:
                    local_bytes = render_local_text_fallback(image_bytes=image_bytes, analysis=localized_analysis)
                    if local_bytes:
                        result = {
                            "engine_id": "local_structured_edit",
                            "motor": "Edição Local Estruturada",
                            "url": _data_uri_from_b64(base64.b64encode(local_bytes).decode("utf-8"), "image/png"),
                            "raw": {"strategy": attempt_plan["reason"]},
                        }

                # --- Fallback: GPT Image 1.5 edit (full image) ---
                if result is None:
                    final_prompt_for_edit = final_prompt + _build_localized_prompt_appendix(localized_analysis, instruction_info)
                    try:
                        result = await _edit_openai_image(
                            client=client,
                            image_bytes=image_bytes,
                            filename=image_filename,
                            content_type=image_content_type,
                            final_prompt=final_prompt_for_edit,
                            aspect_ratio=engine_aspect_ratio,
                            quality=openai_quality,
                            openai_key=openai_key,
                            openai_size=openai_size,
                            mask_bytes=localized_mask,
                            input_fidelity="high",
                        )
                    except Exception as edit_exc:
                        if all_localized_analyses:
                            fallback_bytes = render_all_local_text_replacements(image_bytes=image_bytes, analyses=all_localized_analyses)
                            if not fallback_bytes and localized_analysis:
                                fallback_bytes = render_local_text_fallback(image_bytes=image_bytes, analysis=localized_analysis)
                            if fallback_bytes:
                                result = {
                                    "engine_id": "local_structured_edit",
                                    "motor": "Edição Local Estruturada",
                                    "url": _data_uri_from_b64(base64.b64encode(fallback_bytes).decode("utf-8"), "image/png"),
                                    "raw": {"fallback_reason": str(edit_exc), "strategy": "openai_failed_then_local"},
                                }
                            else:
                                raise
                        else:
                            raise

                result = await _apply_postprocess_if_needed(client, result, requested_dimensions)

                yield _sse({
                    "status": f"Edição concluída com sucesso em {result['motor']}.",
                    "progress": 82,
                    "partial_result": {
                        "engine_id": result["engine_id"],
                        "motor": result["motor"],
                        "url": result["url"],
                    },
                    "localized_mode": localized_mode,
                    "localized_analysis": localized_analysis,
                    "attempt_plan": attempt_plan,
                })

                yield _sse({
                    "status": "Concluído. Entregando a imagem editada.",
                    "progress": 100,
                    "improved_prompt": improved["prompt_final"],
                    "negative_prompt": improved["negative_prompt"],
                    "creative_direction": improved["creative_direction"],
                    "layout_notes": improved["layout_notes"],
                    "preservation_rules": improved["preservation_rules"],
                    "edit_strategy": improved["edit_strategy"],
                    "micro_detail_rules": improved["micro_detail_rules"],
                    "consistency_rules": improved["consistency_rules"],
                    "final_prompt": final_prompt,
                    "localized_mode": localized_mode,
                    "localized_analysis": localized_analysis,
                    "attempt_plan": attempt_plan,
                    "final_results": [
                        {
                            "engine_id": result["engine_id"],
                            "motor": result["motor"],
                            "url": result["url"],
                        }
                    ],
                })

        except Exception as e:
            yield _sse({"error": f"Erro interno no editor: {str(e)}"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
