from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import SQLModel, Field

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

class Robot(SQLModel, table=True):
    __tablename__ = "robot"

    id: Optional[int] = Field(default=None, primary_key=True)
    public_id: str = Field(index=True, unique=True)

    title: str
    description: str = Field(default="")
    avatar_data: Optional[str] = Field(default=None)

    system_instructions: str
    model: str = Field(default="gpt-4o-mini")
    
    # NOVO CAMPO: Guarda o log de arquivos subidos (ex: [{"filename": "doc.pdf", "date": "..."}])
    knowledge_files_json: str = Field(default="[]")

    created_at: datetime = Field(default_factory=utcnow)

class BusinessCore(SQLModel, table=True):
    __tablename__ = "business_core"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    robot_id: int = Field(foreign_key="robot.id", index=True, unique=True)
    
    company_name: str = Field(default="")
    city_state: str = Field(default="")
    service_area: str = Field(default="")
    main_audience: str = Field(default="")
    services_products: str = Field(default="")
    real_differentials: str = Field(default="")
    restrictions: str = Field(default="")
    
    reviews: str = Field(default="")
    testimonials: str = Field(default="")
    usable_links_texts: str = Field(default="")
    forbidden_content: str = Field(default="")
    
    site: str = Field(default="")
    google_business_profile: str = Field(default="")
    instagram: str = Field(default="")
    linkedin: str = Field(default="")
    youtube: str = Field(default="")
    tiktok: str = Field(default="")

    knowledge_text: str = Field(default="") # Armazena todo o texto extraído dos arquivos
    knowledge_files_json: str = Field(default="[]") # Histórico de arquivos enviados
    
    updated_at: datetime = Field(default_factory=utcnow)

class ChatMessage(SQLModel, table=True):
    __tablename__ = "chat_message"

    id: Optional[int] = Field(default=None, primary_key=True)
    robot_id: int = Field(foreign_key="robot.id", index=True)

    role: str
    content: str
    created_at: datetime = Field(default_factory=utcnow)

class CompetitionAnalysis(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    public_id: str = Field(index=True, unique=True)
    instagrams_json: str = Field(default="[]")
    sites_json: str = Field(default="[]")
    company_json: str = Field(default="{}")

    status: str = Field(default="queued", index=True)
    stage: str = Field(default="Na fila")
    progress: float = Field(default=0.0)

    result_json: str | None = Field(default=None)
    error: str | None = Field(default=None)

    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

class AuthorityEdit(SQLModel, table=True):
    __tablename__ = "authority_edit"

    id: Optional[int] = Field(default=None, primary_key=True)
    robot_id: int = Field(foreign_key="robot.id", index=True)

    user_message: str
    assistant_reply: str = Field(default="")
    changes_json: str = Field(default="[]")
    summary: str = Field(default="")

    before_score: int = Field(default=0)
    after_score: int = Field(default=0)

    before_hash: str = Field(index=True)
    after_hash: str = Field(index=True)

    created_at: datetime = Field(default_factory=utcnow)

class AuthorityAgentRun(SQLModel, table=True):
    __tablename__ = "authority_agent_run"

    id: Optional[int] = Field(default=None, primary_key=True)
    client_id: str = Field(index=True)
    agent_key: str = Field(index=True)
    nucleus_json: str
    output_text: str
    created_at: datetime = Field(default_factory=utcnow, index=True)