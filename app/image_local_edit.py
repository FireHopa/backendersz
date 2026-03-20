from __future__ import annotations

import base64
import io
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont, ImageStat

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


DEFAULT_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/Library/Fonts/Arial.ttf",
]


def _parse_json_safe(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        cleaned = (text or "").strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)


def _data_uri_from_bytes(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _hex_or_default(value: Optional[str], default: str) -> str:
    if not value:
        return default
    try:
        ImageColor.getrgb(value)
        return value
    except Exception:
        return default


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = [max(0, min(255, int(x))) for x in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def _norm_box_to_px(box: Optional[Dict[str, Any]], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if not box:
        return None
    try:
        x = float(box.get("x", 0.0))
        y = float(box.get("y", 0.0))
        w = float(box.get("w", 0.0))
        h = float(box.get("h", 0.0))
    except Exception:
        return None

    x1 = _clamp(int(round(x * width)), 0, max(0, width - 1))
    y1 = _clamp(int(round(y * height)), 0, max(0, height - 1))
    x2 = _clamp(int(round((x + w) * width)), 1, width)
    y2 = _clamp(int(round((y + h) * height)), 1, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _px_box_to_norm(rect: Tuple[int, int, int, int], width: int, height: int) -> Dict[str, float]:
    x1, y1, x2, y2 = rect
    return {
        "x": round(max(0.0, min(1.0, x1 / max(1, width))), 6),
        "y": round(max(0.0, min(1.0, y1 / max(1, height))), 6),
        "w": round(max(0.0, min(1.0, (x2 - x1) / max(1, width))), 6),
        "h": round(max(0.0, min(1.0, (y2 - y1) / max(1, height))), 6),
    }


def _inflate_rect(rect: Tuple[int, int, int, int], padx: int, pady: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = rect
    return (
        _clamp(x1 - padx, 0, width),
        _clamp(y1 - pady, 0, height),
        _clamp(x2 + padx, 0, width),
        _clamp(y2 + pady, 0, height),
    )


def _rect_union(rects: Iterable[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    rects = list(rects)
    if not rects:
        return None
    x1 = min(r[0] for r in rects)
    y1 = min(r[1] for r in rects)
    x2 = max(r[2] for r in rects)
    y2 = max(r[3] for r in rects)
    return (x1, y1, x2, y2)


def _rect_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _rect_center_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    acx = (a[0] + a[2]) / 2.0
    acy = (a[1] + a[3]) / 2.0
    bcx = (b[0] + b[2]) / 2.0
    bcy = (b[1] + b[3]) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    preferred: List[str] = []
    if bold:
        preferred.extend([p for p in DEFAULT_FONT_CANDIDATES if "bold" in p.lower() or p.lower().endswith("bd.ttf")])
    preferred.extend(DEFAULT_FONT_CANDIDATES)
    seen = set()
    for path in preferred:
        if path in seen:
            continue
        seen.add(path)
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _extract_text_replacement(instruction: str) -> Optional[Dict[str, str]]:
    text = (instruction or "").strip()
    if not text:
        return None

    patterns = [
        r"troqu[eia]?\s+[\"“”'](?P<old>.+?)[\"“”']\s+por\s+[\"“”'](?P<new>.+?)[\"“”']",
        r"substitu[a-z]*\s+[\"“”'](?P<old>.+?)[\"“”']\s+por\s+[\"“”'](?P<new>.+?)[\"“”']",
        r"alter[eia]?\s+[\"“”'](?P<old>.+?)[\"“”']\s+para\s+[\"“”'](?P<new>.+?)[\"“”']",
        r"mud[eia]?\s+[\"“”'](?P<old>.+?)[\"“”']\s+para\s+[\"“”'](?P<new>.+?)[\"“”']",
        r"troqu[eia]?\s+(?P<old>[^\n]+?)\s+por\s+(?P<new>[^\n]+)",
        r"substitu[a-z]*\s+(?P<old>[^\n]+?)\s+por\s+(?P<new>[^\n]+)",
        r"alter[eia]?\s+(?P<old>[^\n]+?)\s+para\s+(?P<new>[^\n]+)",
        r"mud[eia]?\s+(?P<old>[^\n]+?)\s+para\s+(?P<new>[^\n]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            old = (match.group("old") or "").strip(" '‘’\"“”.,;:")
            new = (match.group("new") or "").strip(" '‘’\"“”.,;:")
            if old and new and old.lower() != new.lower():
                return {"old_text": old, "new_text": new}
    return None


def extract_edit_instruction_info(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    lowered = raw.lower()
    replacement = _extract_text_replacement(raw)

    pure_text_positive_patterns = [
        r"\b(troque|substitua|mude|altere|corrija|edite|atualize)\b.*\b(texto|frase|headline|subheadline|bot[aã]o|cta|data|cidade|local)\b",
        r"\b(troque|substitua|mude|altere|corrija|atualize)\b.+\bpara\b.+",
    ]
    broad_visual_markers = [
        "fundo", "background", "cenário", "cenario", "produto", "pessoa", "rosto", "objeto", "logo", "logotipo",
        "marca", "remove", "remova", "adicione", "adicionar", "insira", "troque a cor", "mudar cor", "cor do",
        "iluminação", "iluminacao", "sombra", "perspectiva", "composição", "composicao", "estilo", "realista",
        "fotorrealista", "avatar", "personagem", "roupa", "embalagem", "cenografia"
    ]
    mentions_broad_visual = any(token in lowered for token in broad_visual_markers)
    positive_match = any(re.search(pattern, raw, flags=re.IGNORECASE) for pattern in pure_text_positive_patterns)

    edit_type = "generic_edit"
    if replacement and (positive_match or not mentions_broad_visual):
        edit_type = "text_replace"

    target_hint = None
    if replacement:
        target_hint = replacement["old_text"]
    else:
        hint_match = re.search(r"\b(bot[aã]o|cta|headline|subheadline|data|cidade|local)\b", raw, flags=re.IGNORECASE)
        if hint_match:
            target_hint = hint_match.group(1)

    return {
        "raw_instruction": raw,
        "edit_type": edit_type,
        "replacement": replacement,
        "target_hint": target_hint,
        "is_pure_text_edit": bool(edit_type == "text_replace" and replacement),
        "mentions_broad_visual_changes": mentions_broad_visual,
    }


def _encode_png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _crop_data_url(image_bytes: bytes, rect: Tuple[int, int, int, int], mime: str = "image/png") -> str:
    with Image.open(io.BytesIO(image_bytes)) as im:
        crop = im.convert("RGBA").crop(rect)
        return _data_uri_from_bytes(_encode_png(crop), mime)


def _local_text_candidates(image_bytes: bytes) -> List[Dict[str, Any]]:
    if cv2 is None:
        return []
    np_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if decoded is None:
        return []

    height, width = decoded.shape[:2]
    gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    candidates: List[Tuple[int, int, int, int]] = []

    try:
        mser = cv2.MSER_create(_min_area=max(40, int(width * height * 0.00002)), _max_area=max(6000, int(width * height * 0.04)))
        regions, _ = mser.detectRegions(gray)
        for pts in regions:
            x, y, w, h = cv2.boundingRect(pts)
            if w < 10 or h < 8:
                continue
            aspect = w / max(1, h)
            area = w * h
            if aspect < 0.7 or aspect > 18:
                continue
            if area < 100 or area > width * height * 0.10:
                continue
            candidates.append((x, y, x + w, y + h))
    except Exception:
        pass

    # Secondary pass for stronger text lines
    for invert in (False, True):
        img = 255 - gray if invert else gray
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
        if not invert:
            thresh = 255 - thresh
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 40), max(3, height // 120)))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 24 or h < 10:
                continue
            if w > width * 0.96 or h > height * 0.30:
                continue
            aspect = w / max(1, h)
            fill_ratio = cv2.contourArea(cnt) / float(max(1, w * h))
            if aspect < 1.6 or fill_ratio < 0.08:
                continue
            candidates.append((x, y, x + w, y + h))

    if not candidates:
        return []

    # Merge close rows into line-level candidates
    candidates.sort(key=lambda r: (r[1], r[0]))
    merged: List[Tuple[int, int, int, int]] = []
    for rect in candidates:
        attached = False
        for idx, existing in enumerate(merged):
            ex1, ey1, ex2, ey2 = existing
            rx1, ry1, rx2, ry2 = rect
            vertical_overlap = max(0, min(ey2, ry2) - max(ey1, ry1))
            min_h = min(ey2 - ey1, ry2 - ry1)
            same_row = vertical_overlap >= max(4, min_h * 0.35)
            near_x = rx1 <= ex2 + max(16, width // 40)
            if same_row and near_x:
                merged[idx] = (min(ex1, rx1), min(ey1, ry1), max(ex2, rx2), max(ey2, ry2))
                attached = True
                break
        if not attached:
            merged.append(rect)

    # Dedupe by IoU and score
    final: List[Tuple[int, int, int, int]] = []
    for rect in sorted(merged, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True):
        if any(_rect_iou(rect, existing) > 0.55 for existing in final):
            continue
        final.append(rect)
        if len(final) >= 18:
            break

    payload: List[Dict[str, Any]] = []
    for rect in final:
        x1, y1, x2, y2 = rect
        w = x2 - x1
        h = y2 - y1
        area = w * h
        density = area / float(max(1, width * height))
        score = min(0.98, 0.42 + min(0.34, w / max(1, width)) + min(0.18, h / max(1, height)) - min(0.14, density))
        payload.append({
            "bbox": _px_box_to_norm(rect, width, height),
            "px_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "score": round(score, 4),
        })
    return payload


def _build_zoom_rect(base_rect: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    bw = base_rect[2] - base_rect[0]
    bh = base_rect[3] - base_rect[1]
    pad_x = max(24, int(bw * 0.45))
    pad_y = max(18, int(bh * 0.85))
    return _inflate_rect(base_rect, pad_x, pad_y, width, height)


def _extract_sub_candidates(candidates: List[Dict[str, Any]], crop_rect: Tuple[int, int, int, int], width: int, height: int) -> List[Dict[str, Any]]:
    cx1, cy1, cx2, cy2 = crop_rect
    selected = []
    for item in candidates:
        px = item.get("px_bbox") or {}
        rect = (int(px.get("x1", 0)), int(px.get("y1", 0)), int(px.get("x2", 0)), int(px.get("y2", 0)))
        if rect[2] <= cx1 or rect[0] >= cx2 or rect[3] <= cy1 or rect[1] >= cy2:
            continue
        clipped = (max(cx1, rect[0]), max(cy1, rect[1]), min(cx2, rect[2]), min(cy2, rect[3]))
        selected.append({
            "bbox": _px_box_to_norm(clipped, width, height),
            "score": item.get("score", 0.5),
        })
    selected.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return selected[:10]


def _sanitize_analysis(data: Dict[str, Any], replacement: Dict[str, str], width: int, height: int) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    try:
        confidence = float(data.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0

    operation = data.get("operation") or "text_replace"
    style = data.get("style") or {}

    bbox = _norm_box_to_px(data.get("bbox"), width, height)
    text_bbox = _norm_box_to_px(data.get("text_bbox"), width, height)
    container_bbox = _norm_box_to_px(data.get("container_bbox"), width, height)

    if bbox is None and text_bbox is not None:
        bbox = _inflate_rect(text_bbox, 8, 8, width, height)
    if text_bbox is None and bbox is not None:
        text_bbox = bbox

    if not bbox and not text_bbox and not container_bbox:
        return None

    payload: Dict[str, Any] = {
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "operation": operation,
        "target_text": replacement["old_text"],
        "replacement_text": replacement["new_text"],
        "reason": data.get("reason") or "localized-detection",
        "style": {
            "alignment": (style.get("alignment") or "center").lower(),
            "text_color": style.get("text_color"),
            "background_color": style.get("background_color"),
            "border_color": style.get("border_color"),
            "border_radius": float(style.get("border_radius") or 0.0),
            "shadow": bool(style.get("shadow", False)),
            "glow": bool(style.get("glow", False)),
            "font_weight": (style.get("font_weight") or "regular").lower(),
        },
    }
    if bbox:
        payload["bbox"] = _px_box_to_norm(bbox, width, height)
    if text_bbox:
        payload["text_bbox"] = _px_box_to_norm(text_bbox, width, height)
    if container_bbox:
        payload["container_bbox"] = _px_box_to_norm(container_bbox, width, height)
    return payload


def _merge_analysis_with_candidates(
    analysis: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    width: int,
    height: int,
) -> Dict[str, Any]:
    bbox = _norm_box_to_px(analysis.get("bbox"), width, height)
    text_bbox = _norm_box_to_px(analysis.get("text_bbox"), width, height)
    container_bbox = _norm_box_to_px(analysis.get("container_bbox"), width, height)
    anchor = text_bbox or bbox or container_bbox
    if not anchor:
        return analysis

    best_rect = None
    best_score = -999.0
    for item in candidates:
        px = item.get("px_bbox") or {}
        rect = (int(px.get("x1", 0)), int(px.get("y1", 0)), int(px.get("x2", 0)), int(px.get("y2", 0)))
        if rect[2] <= rect[0] or rect[3] <= rect[1]:
            continue
        iou = _rect_iou(anchor, rect)
        dist = _rect_center_distance(anchor, rect)
        diag = math.hypot(width, height)
        proximity = 1.0 - min(1.0, dist / max(1.0, diag * 0.18))
        candidate_score = float(item.get("score", 0.5) or 0.5)
        score = iou * 1.6 + proximity * 0.8 + candidate_score * 0.35
        if score > best_score:
            best_rect = rect
            best_score = score

    if best_rect is None:
        return analysis

    merged_text = _rect_union([r for r in [anchor, best_rect] if r])
    if merged_text:
        analysis["text_bbox"] = _px_box_to_norm(merged_text, width, height)
        analysis["bbox"] = _px_box_to_norm(_inflate_rect(merged_text, 8, 8, width, height), width, height)

    if analysis.get("operation") == "button_text_replace":
        container = container_bbox or _inflate_rect(merged_text, 24, 16, width, height)
        analysis["container_bbox"] = _px_box_to_norm(container, width, height)

    analysis["confidence"] = round(min(0.99, max(float(analysis.get("confidence", 0.0) or 0.0), 0.56 + max(0.0, best_score * 0.08))), 4)
    analysis["candidate_refined"] = True
    return analysis


def _sample_region_mean(image: Image.Image, rect: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    crop = image.crop(rect).convert("RGB")
    stat = ImageStat.Stat(crop)
    vals = stat.mean[:3]
    return tuple(int(v) for v in vals)


def _estimate_style_from_pixels(
    image_bytes: bytes,
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    with Image.open(io.BytesIO(image_bytes)) as im:
        image = im.convert("RGBA")
        width, height = image.size
        style = dict(analysis.get("style") or {})

        text_rect = _norm_box_to_px(analysis.get("text_bbox"), width, height)
        container_rect = _norm_box_to_px(analysis.get("container_bbox"), width, height)
        bbox = _norm_box_to_px(analysis.get("bbox"), width, height)
        anchor = text_rect or bbox

        if anchor:
            tx1, ty1, tx2, ty2 = anchor
            inner = image.crop(anchor).convert("RGB")
            arr = np.array(inner)
            if arr.size:
                luminance = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
                flat = arr.reshape(-1, 3)
                top_n = max(8, len(flat) // 7)
                dark = flat[np.argsort(luminance.reshape(-1))[:top_n]]
                light = flat[np.argsort(luminance.reshape(-1))[-top_n:]]
                dark_mean = tuple(int(x) for x in dark.mean(axis=0)) if len(dark) else (20, 20, 20)
                light_mean = tuple(int(x) for x in light.mean(axis=0)) if len(light) else (235, 235, 235)

                if not style.get("text_color"):
                    outer_rect = _inflate_rect(anchor, max(4, (tx2 - tx1) // 6), max(4, (ty2 - ty1) // 4), width, height)
                    outer_mean = _sample_region_mean(image, outer_rect)
                    outer_l = sum(outer_mean)
                    dark_l = sum(dark_mean)
                    light_l = sum(light_mean)
                    pick = light_mean if abs(light_l - outer_l) > abs(dark_l - outer_l) else dark_mean
                    style["text_color"] = _rgb_to_hex(pick)

                if "glow" not in style:
                    style["glow"] = abs(sum(light_mean) - sum(dark_mean)) > 360 and max(abs(light_mean[i] - dark_mean[i]) for i in range(3)) > 90

        if container_rect:
            if not style.get("background_color"):
                bg = _sample_region_mean(image, container_rect)
                style["background_color"] = _rgb_to_hex(bg)
            if not style.get("border_radius"):
                ch = container_rect[3] - container_rect[1]
                style["border_radius"] = round(max(8.0, ch * 0.24), 2)

        style.setdefault("alignment", "center")
        style.setdefault("font_weight", "bold" if analysis.get("operation") == "button_text_replace" else "regular")
        style.setdefault("shadow", False)
        style.setdefault("glow", False)
        style.setdefault("text_color", "#FFFFFF" if analysis.get("operation") == "button_text_replace" else "#111111")
        analysis["style"] = style
        return analysis


async def _ask_openai_for_json(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": messages,
        "temperature": 0.0,
    }
    resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return _parse_json_safe(content)


async def analyze_region_with_openai(
    client: httpx.AsyncClient,
    image_bytes: bytes,
    content_type: str,
    instruction: str,
    model: str,
    api_key: str,
) -> Optional[Dict[str, Any]]:
    replacement = _extract_text_replacement(instruction)
    if not replacement:
        return None

    with Image.open(io.BytesIO(image_bytes)) as im:
        width, height = im.size

    candidates = _local_text_candidates(image_bytes)
    candidate_summary = [
        {"bbox": item["bbox"], "score": item["score"]}
        for item in candidates[:12]
    ]

    image_data_url = _data_uri_from_bytes(image_bytes, content_type)
    system_text = """
Você é um analista sênior de edição localizada para interfaces, anúncios e criativos.
Sua tarefa é localizar COM PRECISÃO a região mínima que precisa ser editada quando o usuário quer trocar um texto por outro.

Retorne SOMENTE JSON válido com esta estrutura:
{
  "confidence": number,
  "operation": "text_replace" | "button_text_replace",
  "target_text": string,
  "replacement_text": string,
  "bbox": {"x": number, "y": number, "w": number, "h": number},
  "text_bbox": {"x": number, "y": number, "w": number, "h": number},
  "container_bbox": {"x": number, "y": number, "w": number, "h": number} | null,
  "style": {
    "alignment": "left" | "center" | "right",
    "text_color": string,
    "background_color": string | null,
    "border_color": string | null,
    "border_radius": number,
    "shadow": boolean,
    "glow": boolean,
    "font_weight": "regular" | "medium" | "semibold" | "bold"
  },
  "reason": string
}

Regras:
1. Coordenadas normalizadas entre 0 e 1.
2. bbox deve cobrir a região mínima segura para edição.
3. text_bbox deve cobrir somente o texto.
4. Se o texto estiver dentro de botão, badge ou chip, preencha container_bbox com a área completa do elemento e use operation=button_text_replace.
5. Priorize precisão e preservação do restante da imagem.
6. Se houver candidatos locais fornecidos, use-os como reforço de precisão. Você pode ignorar candidatos ruins.
7. Mesmo que o texto esteja pequeno, borrado, parcialmente cortado ou com leve diferença visual, localize a região semanticamente correta.
8. Se houver mais de um bloco parecido, escolha o que melhor combina com a instrução do usuário e com a hierarquia visual da peça.
9. Evite caixas excessivamente grandes. Prefira a menor área segura que permita editar sem afetar o restante.
"""

    user_text = (
        f"Instrução do usuário: {instruction}\n"
        f"Texto atual esperado: {replacement['old_text']}\n"
        f"Novo texto esperado: {replacement['new_text']}\n"
        f"Dimensões da imagem: {width}x{height}\n"
        f"Candidatos locais detectados por visão computacional: {json.dumps(candidate_summary, ensure_ascii=False)}\n"
        "Localize a região exata. Considere a semântica do pedido, a hierarquia visual e os candidatos detectados. Estime também o estilo visual necessário para uma edição localizada profissional."
    )

    base_messages = [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]

    first_pass = _sanitize_analysis(await _ask_openai_for_json(client, api_key, model, base_messages), replacement, width, height)
    if not first_pass:
        return None

    first_pass = _merge_analysis_with_candidates(first_pass, candidates, width, height)

    anchor_rect = _norm_box_to_px(first_pass.get("container_bbox") or first_pass.get("bbox") or first_pass.get("text_bbox"), width, height)
    if anchor_rect:
        zoom_rect = _build_zoom_rect(anchor_rect, width, height)
        zoom_candidates = _extract_sub_candidates(candidates, zoom_rect, width, height)
        zoom_data_url = _crop_data_url(image_bytes, zoom_rect, mime="image/png")
        zoom_summary = json.dumps({"crop_bbox": _px_box_to_norm(zoom_rect, width, height), "sub_candidates": zoom_candidates}, ensure_ascii=False)
        refine_system = system_text + "\nRefine a detecção usando a imagem recortada em alta proximidade."
        refine_text = (
            f"Primeira estimativa: {json.dumps(first_pass, ensure_ascii=False)}\n"
            f"Contexto do recorte: {zoom_summary}\n"
            "A imagem anexada agora é um recorte aproximado da região alvo.\n"
            "Responda novamente em coordenadas NORMALIZADAS DA IMAGEM ORIGINAL, com a caixa mais precisa possível.\n"
            "Se perceber que o texto exato não está no recorte, mantenha a melhor estimativa original e reduza a confiança."
        )
        refined = _sanitize_analysis(
            await _ask_openai_for_json(
                client,
                api_key,
                model,
                [
                    {"role": "system", "content": refine_system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": refine_text},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                            {"type": "image_url", "image_url": {"url": zoom_data_url}},
                        ],
                    },
                ],
            ),
            replacement,
            width,
            height,
        )
        if refined:
            refined = _merge_analysis_with_candidates(refined, candidates, width, height)
            first_conf = float(first_pass.get("confidence", 0.0) or 0.0)
            refined_conf = float(refined.get("confidence", 0.0) or 0.0)
            if refined_conf >= first_conf - 0.03:
                first_pass = refined
                first_pass["refined_pass"] = True

    first_pass["candidate_count"] = len(candidates)
    first_pass["detection_mode"] = "two_pass_localized"
    first_pass = _estimate_style_from_pixels(image_bytes, first_pass)
    return first_pass


def should_use_localized_edit(analysis: Optional[Dict[str, Any]]) -> bool:
    if not analysis:
        return False
    confidence = float(analysis.get("confidence", 0.0) or 0.0)
    text_bbox = analysis.get("text_bbox")
    bbox = analysis.get("bbox")
    mode = analysis.get("detection_mode")
    refined = bool(analysis.get("candidate_refined") or analysis.get("refined_pass"))
    if confidence >= 0.58:
        return bool(text_bbox or bbox)
    if confidence >= 0.48 and refined and mode == "two_pass_localized":
        return bool(text_bbox or bbox)
    return False


def should_use_local_text_render(
    analysis: Optional[Dict[str, Any]],
    instruction_info: Optional[Dict[str, Any]] = None,
) -> bool:
    if not analysis:
        return False
    instruction_info = instruction_info or {}
    if not instruction_info.get("is_pure_text_edit"):
        return False
    confidence = float(analysis.get("confidence", 0.0) or 0.0)
    operation = (analysis.get("operation") or "").lower()
    text_bbox = analysis.get("text_bbox") or analysis.get("bbox")
    if operation not in {"text_replace", "button_text_replace"}:
        return False
    if not text_bbox:
        return False
    return confidence >= 0.58


def build_mask_from_analysis(
    image_bytes: bytes,
    analysis: Dict[str, Any],
) -> Optional[bytes]:
    with Image.open(io.BytesIO(image_bytes)) as im:
        rgba = im.convert("RGBA")
        width, height = rgba.size

        base_rect = _norm_box_to_px(analysis.get("bbox"), width, height)
        text_rect = _norm_box_to_px(analysis.get("text_bbox"), width, height)
        container_rect = _norm_box_to_px(analysis.get("container_bbox"), width, height)
        operation = analysis.get("operation") or "text_replace"
        edit_rect = text_rect or base_rect or container_rect
        if not edit_rect:
            return None

        if operation == "button_text_replace" and container_rect and text_rect:
            bw = container_rect[2] - container_rect[0]
            pad_x = max(12, int(bw * 0.09))
            pad_y = max(8, int((text_rect[3] - text_rect[1]) * 0.55))
            edit_rect = _inflate_rect(text_rect, pad_x, pad_y, width, height)
        elif text_rect:
            pad_x = max(8, int((text_rect[2] - text_rect[0]) * 0.30))
            pad_y = max(8, int((text_rect[3] - text_rect[1]) * 0.50))
            edit_rect = _inflate_rect(text_rect, pad_x, pad_y, width, height)
        else:
            pad_x = max(12, int((edit_rect[2] - edit_rect[0]) * 0.16))
            pad_y = max(10, int((edit_rect[3] - edit_rect[1]) * 0.24))
            edit_rect = _inflate_rect(edit_rect, pad_x, pad_y, width, height)

        mask = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(mask)
        radius = int(float(analysis.get("style", {}).get("border_radius") or 0.0))
        draw.rounded_rectangle(edit_rect, radius=max(6, radius // 2 if radius else 10), fill=(0, 0, 0, 0))

        out = io.BytesIO()
        mask.save(out, format="PNG")
        return out.getvalue()


def _inpaint_rgba(
    image: Image.Image,
    rect: Tuple[int, int, int, int],
    radius: int = 3,
) -> Image.Image:
    if cv2 is None:
        return image
    width, height = image.size
    np_img = np.array(image.convert("RGBA"))
    alpha = np_img[:, :, 3]
    rgb = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = rect
    mask[y1:y2, x1:x2] = 255
    inpainted_rgb = cv2.inpaint(rgb, mask, radius, cv2.INPAINT_TELEA)
    merged = np.dstack([inpainted_rgb, alpha])
    return Image.fromarray(merged, mode="RGBA")


def _fit_text(
    text: str,
    box: Tuple[int, int, int, int],
    font_weight: str,
    multiline: bool = False,
) -> Tuple[ImageFont.ImageFont, Tuple[int, int, int, int], List[str], int]:
    bw = max(1, box[2] - box[0])
    bh = max(1, box[3] - box[1])
    bold = font_weight in {"semibold", "bold", "heavy"}
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))

    def wrap(candidate_font: ImageFont.ImageFont) -> Tuple[List[str], Tuple[int, int, int, int]]:
        if not multiline:
            bounds = probe.multiline_textbbox((0, 0), text, font=candidate_font, spacing=0, align="left")
            return [text], bounds
        words = text.split()
        if not words:
            bounds = probe.textbbox((0, 0), text, font=candidate_font)
            return [text], bounds
        lines = [words[0]]
        for word in words[1:]:
            trial = f"{lines[-1]} {word}".strip()
            trial_bounds = probe.textbbox((0, 0), trial, font=candidate_font)
            if (trial_bounds[2] - trial_bounds[0]) <= bw * 0.88:
                lines[-1] = trial
            else:
                lines.append(word)
        bounds = probe.multiline_textbbox((0, 0), "\n".join(lines), font=candidate_font, spacing=max(2, int(candidate_font.size * 0.14)), align="left")
        return lines, bounds

    min_size = max(12, int(bh * 0.18))
    max_size = max(min_size, int(bh * (0.92 if not multiline else 0.82)))
    best_font: ImageFont.ImageFont = _load_font(min_size, bold=bold)
    best_bounds = probe.textbbox((0, 0), text, font=best_font)
    best_lines = [text]
    best_spacing = max(2, int(best_font.size * 0.12)) if hasattr(best_font, 'size') else 2

    lo, hi = min_size, max_size
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = _load_font(mid, bold=bold)
        lines, bounds = wrap(candidate)
        tw = bounds[2] - bounds[0]
        th = bounds[3] - bounds[1]
        spacing = max(2, int(getattr(candidate, 'size', mid) * 0.12))
        fits = tw <= bw * 0.94 and th <= bh * 0.90 and len(lines) <= (3 if multiline else 1)
        if fits:
            best_font = candidate
            best_bounds = bounds
            best_lines = lines
            best_spacing = spacing
            lo = mid + 1
        else:
            hi = mid - 1

    return best_font, best_bounds, best_lines, best_spacing


def _draw_text_with_effects(
    overlay: Image.Image,
    text: str,
    box: Tuple[int, int, int, int],
    style: Dict[str, Any],
    multiline: bool,
) -> None:
    font, bounds, lines, spacing = _fit_text(text, box, style.get("font_weight", "regular"), multiline=multiline)
    draw = ImageDraw.Draw(overlay)
    content = "\n".join(lines)
    tw = bounds[2] - bounds[0]
    th = bounds[3] - bounds[1]
    bw = box[2] - box[0]
    bh = box[3] - box[1]
    align = (style.get("alignment") or "center").lower()

    if align == "left":
        tx = box[0] + int(bw * 0.06) - bounds[0]
        text_align = "left"
    elif align == "right":
        tx = box[2] - tw - int(bw * 0.06) - bounds[0]
        text_align = "right"
    else:
        tx = box[0] + (bw - tw) / 2 - bounds[0]
        text_align = "center"
    ty = box[1] + (bh - th) / 2 - bounds[1]

    color = _hex_or_default(style.get("text_color"), "#FFFFFF")
    stroke_width = 1 if style.get("font_weight") in {"semibold", "bold"} else 0
    stroke_fill = None

    if style.get("glow"):
        glow = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow)
        glow_draw.multiline_text((tx, ty), content, font=font, fill=(255, 255, 255, 150), spacing=spacing, align=text_align)
        glow = glow.filter(ImageFilter.GaussianBlur(radius=max(4, int(getattr(font, 'size', 24) * 0.14))))
        overlay.alpha_composite(glow)
        draw = ImageDraw.Draw(overlay)

    if style.get("shadow"):
        shadow_alpha = 120 if not style.get("glow") else 70
        draw.multiline_text((tx + 2, ty + 2), content, font=font, fill=(0, 0, 0, shadow_alpha), spacing=spacing, align=text_align)

    draw.multiline_text(
        (tx, ty),
        content,
        font=font,
        fill=color,
        spacing=spacing,
        align=text_align,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )


def render_local_text_fallback(
    image_bytes: bytes,
    analysis: Dict[str, Any],
) -> Optional[bytes]:
    with Image.open(io.BytesIO(image_bytes)) as im:
        image = im.convert("RGBA")
        width, height = image.size

        text = (analysis.get("replacement_text") or "").strip()
        if not text:
            return None

        style = analysis.get("style") or {}
        operation = analysis.get("operation") or "text_replace"

        bbox = _norm_box_to_px(analysis.get("bbox"), width, height)
        text_rect = _norm_box_to_px(analysis.get("text_bbox"), width, height)
        container_rect = _norm_box_to_px(analysis.get("container_bbox"), width, height)
        if not (bbox or text_rect or container_rect):
            return None

        # Preserve the original artwork whenever possible.
        if text_rect:
            clean_rect = _inflate_rect(text_rect, max(6, (text_rect[2] - text_rect[0]) // 8), max(6, (text_rect[3] - text_rect[1]) // 5), width, height)
            image = _inpaint_rgba(image, clean_rect, radius=3)
        elif bbox:
            image = _inpaint_rgba(image, _inflate_rect(bbox, 4, 4, width, height), radius=3)

        # When the container exists, use it only as a layout box. Do not repaint the whole button.
        if operation == "button_text_replace" and container_rect:
            text_box = container_rect
        else:
            text_box = text_rect or bbox or container_rect
        if not text_box:
            return None

        # Improve typography using measured layout and multiline fallback for longer replacements.
        multiline = len(text) > 14 or (text_box[2] - text_box[0]) / max(1, text_box[3] - text_box[1]) < 3.6
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        _draw_text_with_effects(overlay, text, text_box, style, multiline=multiline)
        image.alpha_composite(overlay)

        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()
