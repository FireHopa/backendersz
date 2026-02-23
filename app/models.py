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

    # Perfil (UI)
    title: str
    description: str = Field(default="")
    avatar_data: Optional[str] = Field(default=None)  # data URL (base64) ou URL

    # Cérebro
    system_instructions: str
    model: str = Field(default="gpt-4o-mini")

    created_at: datetime = Field(default_factory=utcnow)


class ChatMessage(SQLModel, table=True):
    __tablename__ = "chat_message"

    id: Optional[int] = Field(default=None, primary_key=True)
    robot_id: int = Field(foreign_key="robot.id", index=True)

    role: str  # "user" | "assistant"
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

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)



class AuthorityEdit(SQLModel, table=True):
    __tablename__ = "authority_edit"

    id: Optional[int] = Field(default=None, primary_key=True)
    robot_id: int = Field(foreign_key="robot.id", index=True)

    # Audit trail
    user_message: str
    assistant_reply: str = Field(default="")
    changes_json: str = Field(default="[]")   # list[dict]
    summary: str = Field(default="")

    before_score: int = Field(default=0)
    after_score: int = Field(default=0)

    # Fingerprints to prevent duplicate edits
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
