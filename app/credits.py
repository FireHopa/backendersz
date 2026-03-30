from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, Response
from sqlmodel import Session

from .models import User

INITIAL_CREDITS = 12_000
DAILY_FREE_CREDITS = 3_000

CREDIT_HEADER = "X-User-Credits"
CREDIT_CHARGED_HEADER = "X-Credits-Charged"
CREDIT_ACTION_HEADER = "X-Credit-Action"


@dataclass(frozen=True)
class CreditAction:
    key: str
    title: str
    description: str
    credits: int
    category: str


@dataclass(frozen=True)
class CreditPlan:
    id: str
    title: str
    description: str
    monthly_fit: str
    display_price: str
    base_credits: int
    bonus_credits: int
    badge: str | None = None
    recommended: bool = False

    @property
    def total_credits(self) -> int:
        return int(self.base_credits + self.bonus_credits)


_ACTIONS: tuple[CreditAction, ...] = (
    CreditAction(
        key="robot_create",
        title="Criar robô por briefing",
        description="Gera o robô inicial com estratégia, estrutura e prompt-base.",
        credits=1_800,
        category="Robôs",
    ),
    CreditAction(
        key="robot_chat_message",
        title="Chat com robô",
        description="Mensagem textual em agentes de chat que não são de autoridade.",
        credits=350,
        category="Robôs",
    ),
    CreditAction(
        key="robot_audio_message",
        title="Chat com robô por áudio",
        description="Transcrição + resposta do agente.",
        credits=700,
        category="Robôs",
    ),
    CreditAction(
        key="authority_assistant_edit",
        title="Assistente de autoridade",
        description="Refina e reescreve instruções do robô com IA.",
        credits=1_200,
        category="Autoridade",
    ),
    CreditAction(
        key="authority_agent_run",
        title="Executar agente de autoridade",
        description="Execução completa de tarefa de autoridade.",
        credits=2_500,
        category="Autoridade",
    ),
    CreditAction(
        key="authority_agent_theme_suggestion",
        title="Gerar temas com IA",
        description="Sugestões de temas para tarefas de autoridade.",
        credits=900,
        category="Autoridade",
    ),
    CreditAction(
        key="authority_agent_video_format_suggestion",
        title="Analisar formato de vídeo",
        description="Recomendação automática do melhor formato para o tema.",
        credits=700,
        category="Autoridade",
    ),
    CreditAction(
        key="competition_find_competitors",
        title="Encontrar concorrentes",
        description="Mapeamento inicial de concorrentes com IA.",
        credits=1_600,
        category="Concorrência",
    ),
    CreditAction(
        key="competition_analyze",
        title="Análise competitiva",
        description="Relatório consolidado de concorrência.",
        credits=4_200,
        category="Concorrência",
    ),
    CreditAction(
        key="skybob_preflight",
        title="SkyBob preflight",
        description="Leitura e estruturação do catálogo antes da missão principal.",
        credits=1_500,
        category="SkyBob",
    ),
    CreditAction(
        key="skybob_full_run",
        title="SkyBob missão completa",
        description="Estudo principal com hooks, insights, calendário e cards.",
        credits=6_500,
        category="SkyBob",
    ),
    CreditAction(
        key="skybob_refine_run",
        title="SkyBob nova rodada de hooks",
        description="Rodada incremental baseada em feedback do usuário.",
        credits=2_800,
        category="SkyBob",
    ),
    CreditAction(
        key="image_generate_from_scratch",
        title="Motor de imagem do zero",
        description="Geração completa de peça visual nas engines configuradas.",
        credits=4_800,
        category="Imagem",
    ),
    CreditAction(
        key="image_edit",
        title="Motor de imagem por edição",
        description="Edição local/IA em imagem de referência.",
        credits=4_200,
        category="Imagem",
    ),
)

_ACTIONS_BY_KEY = {item.key: item for item in _ACTIONS}

_PLANS: tuple[CreditPlan, ...] = (
    CreditPlan(
        id="starter_boost",
        title="Starter Boost",
        description="Entrada rápida para quem usa chat, ajustes e algumas execuções pesadas.",
        monthly_fit="Uso leve a moderado",
        display_price="R$ 39",
        base_credits=12_000,
        bonus_credits=3_000,
    ),
    CreditPlan(
        id="growth_stack",
        title="Growth Stack",
        description="Volume mais equilibrado para rodar autoridade, SkyBob e imagem sem travar rápido.",
        monthly_fit="Uso constante",
        display_price="R$ 89",
        base_credits=32_000,
        bonus_credits=8_000,
        badge="Mais escolhido",
        recommended=True,
    ),
    CreditPlan(
        id="scale_ops",
        title="Scale Ops",
        description="Pacote robusto para operação diária com várias rodadas e geração criativa recorrente.",
        monthly_fit="Uso intenso",
        display_price="R$ 179",
        base_credits=72_000,
        bonus_credits=18_000,
        badge="Melhor custo",
    ),
    CreditPlan(
        id="elite_orbit",
        title="Elite Orbit",
        description="Reserva alta para times ou operação pesada com imagem e SkyBob recorrentes.",
        monthly_fit="Uso extremo",
        display_price="R$ 349",
        base_credits=160_000,
        bonus_credits=40_000,
        badge="Escala máxima",
    ),
)

_PLANS_BY_ID = {item.id: item for item in _PLANS}


def format_credit_amount(value: int | float | None) -> str:
    normalized = int(value or 0)
    return f"{normalized:,}".replace(",", ".")


def get_action_cost(action_key: str) -> int:
    return get_credit_action(action_key).credits


def get_credit_action(action_key: str) -> CreditAction:
    item = _ACTIONS_BY_KEY.get(action_key)
    if not item:
        raise KeyError(f"Ação de crédito desconhecida: {action_key}")
    return item


def get_credit_plan(plan_id: str) -> CreditPlan:
    item = _PLANS_BY_ID.get(plan_id)
    if not item:
        raise KeyError(f"Plano de crédito desconhecido: {plan_id}")
    return item


def list_credit_actions() -> list[dict[str, Any]]:
    return [asdict(item) for item in _ACTIONS]


def list_credit_plans() -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for plan in _PLANS:
        data = asdict(plan)
        data["total_credits"] = plan.total_credits
        payload.append(data)
    return payload


def build_credit_catalog_payload(user: User) -> dict[str, Any]:
    return {
        "current_credits": int(user.credits or 0),
        "daily_free_credits": DAILY_FREE_CREDITS,
        "initial_credits": INITIAL_CREDITS,
        "actions": list_credit_actions(),
        "plans": list_credit_plans(),
    }


def apply_daily_credit_allowance(user: User, session: Session) -> User:
    now = datetime.now(timezone.utc)
    last_reset = user.last_credit_reset or now
    if last_reset.date() >= now.date():
        return user

    user.credits = int(user.credits or 0) + DAILY_FREE_CREDITS
    user.last_credit_reset = now
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def ensure_credits(user: User, action_key: str) -> CreditAction:
    action = get_credit_action(action_key)
    available = int(user.credits or 0)
    if available < action.credits:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Créditos insuficientes. "
                f"{action.title} custa {format_credit_amount(action.credits)} créditos "
                f"e a sua conta tem {format_credit_amount(available)}."
            ),
        )
    return action


def charge_credits(session: Session, user: User, action_key: str) -> CreditAction:
    action = ensure_credits(user, action_key)
    user.credits = max(0, int(user.credits or 0) - int(action.credits))
    session.add(user)
    session.commit()
    session.refresh(user)
    return action


def add_plan_credits(session: Session, user: User, plan_id: str) -> tuple[CreditPlan, int]:
    plan = get_credit_plan(plan_id)
    user.credits = int(user.credits or 0) + int(plan.total_credits)
    session.add(user)
    session.commit()
    session.refresh(user)
    return plan, int(user.credits or 0)


def attach_credit_headers(
    response: Response | None,
    user: User,
    *,
    charged_credits: int = 0,
    action_key: str | None = None,
) -> None:
    if response is None:
        return

    response.headers[CREDIT_HEADER] = str(int(user.credits or 0))
    response.headers[CREDIT_CHARGED_HEADER] = str(max(0, int(charged_credits or 0)))
    if action_key:
        response.headers[CREDIT_ACTION_HEADER] = action_key
