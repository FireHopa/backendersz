from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class BriefingIn(BaseModel):
    company_name: str
    niche: str
    audience: str
    offer: str
    region: str = "Brasil"
    tone: str = "Profissional, direto, claro"
    competitors: Optional[str] = ""
    goals: str = "Aumentar autoridade e ser citado por IA"


class RobotOut(BaseModel):
    public_id: str
    title: str
    description: str = ""
    avatar_data: Optional[str] = None
    created_at: str


class RobotDetail(BaseModel):
    public_id: str
    title: str
    description: str = ""
    avatar_data: Optional[str] = None
    system_instructions: str
    created_at: str


class RobotUpdateIn(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    avatar_data: Optional[str] = None
    system_instructions: Optional[str] = None


class ChatIn(BaseModel):
    message: str
    use_web: bool = False
    web_max_results: int = Field(default=5, ge=1, le=20)
    web_allowed_domains: Optional[list[str]] = None


class ChatMessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: str


class MessageUpdateIn(BaseModel):
    content: str = Field(min_length=1)

class AuthorityAssistantIn(BaseModel):
    message: str = Field(min_length=1)
    history: Optional[list[dict]] = None


class AuthorityAssistantOut(BaseModel):
    apply_change: bool
    before_score: int
    after_score: int
    criteria: list[dict]
    changes_made: list[dict]
    suggestions: list[dict]
    updated_system_instructions: Optional[str] = None
    assistant_reply: str

class AuthorityEditOut(BaseModel):
    id: int
    created_at: str
    user_message: str
    summary: str = ""
    changes_made: list[dict] = []
    before_score: int = 0
    after_score: int = 0



# -----------------------------
# Competition module (PT-BR)
# -----------------------------

class CompetitionBriefingPT(BaseModel):
    """Briefing em português (campos esperados pelo frontend)."""
    nome_empresa: Optional[str] = None
    cidade_estado: str
    bairro: Optional[str] = None
    segmento: str
    servicos: str
    publico_alvo: str
    regiao_atendimento: str
    diferencial: Optional[str] = None
    nivel_preco: Optional[str] = None  # popular|medio|premium
    objetivo: Optional[str] = None
    instagram: Optional[str] = None
    site: Optional[str] = None
    palavras_marca: Optional[list[str]] = None

class CompetitionFindRequest(BaseModel):
    briefing: CompetitionBriefingPT

class CompetitionAnalyzeRequest(BaseModel):
    instagrams: list[str] = []
    sites: list[str] = []
    briefing: Optional[CompetitionBriefingPT] = None

class CompetitionJobV2Out(BaseModel):
    job_id: str
    report_id: Optional[str] = None
    status: str
    stage: Optional[str] = None
    progress: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    error: Optional[str] = None
    warning: Optional[str] = None

class CompetitionReportV2Out(BaseModel):
    report_id: str
    status: str
    result: dict
    updated_at: Optional[str] = None





class CompetitorSuggestion(BaseModel):
    name: str
    website_url: Optional[str] = None
    instagram: Optional[str] = None
    reason: Optional[str] = None
    confidence: Optional[float] = None

class CompetitionFindOut(BaseModel):
    suggestions: list[CompetitorSuggestion] = []
    sources: list[dict] = []
    note: Optional[str] = None
    data_quality: Optional[str] = None



class AuthorityAgentRunIn(BaseModel):
    client_id: str = Field(..., description="Identificador do cliente (armazenado no localStorage).")
    agent_key: str
    nucleus: dict


class AuthorityAgentRunOut(BaseModel):
    id: int
    agent_key: str
    output_text: str
    created_at: str


class AuthorityAgentHistoryOut(BaseModel):
    items: list[AuthorityAgentRunOut]
