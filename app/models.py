from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import SQLModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(SQLModel, table=True):
    __tablename__ = "user"

    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    full_name: Optional[str] = Field(default=None)

    hashed_password: Optional[str] = Field(default=None)
    google_id: Optional[str] = Field(default=None, index=True)

    credits: int = Field(default=100)
    last_credit_reset: datetime = Field(default_factory=utcnow)

    linkedin_access_token: Optional[str] = Field(default=None)
    linkedin_token_expires_at: Optional[datetime] = Field(default=None)
    linkedin_urn: Optional[str] = Field(default=None)

    instagram_meta_access_token: Optional[str] = Field(default=None)
    instagram_meta_token_expires_at: Optional[datetime] = Field(default=None)
    instagram_account_id: Optional[str] = Field(default=None)
    instagram_page_id: Optional[str] = Field(default=None)
    instagram_username: Optional[str] = Field(default=None)

    facebook_user_access_token: Optional[str] = Field(default=None)
    facebook_user_token_expires_at: Optional[datetime] = Field(default=None)
    facebook_page_id: Optional[str] = Field(default=None)
    facebook_page_name: Optional[str] = Field(default=None)
    facebook_page_username: Optional[str] = Field(default=None)
    facebook_page_access_token: Optional[str] = Field(default=None)
    facebook_pages_json: Optional[str] = Field(default="[]")

    youtube_access_token: Optional[str] = Field(default=None)
    youtube_refresh_token: Optional[str] = Field(default=None)
    youtube_token_expires_at: Optional[datetime] = Field(default=None)
    youtube_channel_id: Optional[str] = Field(default=None)
    youtube_channel_title: Optional[str] = Field(default=None)
    youtube_channel_handle: Optional[str] = Field(default=None)
    youtube_channel_thumbnail: Optional[str] = Field(default=None)

    tiktok_access_token: Optional[str] = Field(default=None)
    tiktok_refresh_token: Optional[str] = Field(default=None)
    tiktok_token_expires_at: Optional[datetime] = Field(default=None)
    tiktok_refresh_token_expires_at: Optional[datetime] = Field(default=None)
    tiktok_open_id: Optional[str] = Field(default=None)
    tiktok_scope: Optional[str] = Field(default=None)
    tiktok_display_name: Optional[str] = Field(default=None)
    tiktok_username: Optional[str] = Field(default=None)
    tiktok_avatar_url: Optional[str] = Field(default=None)
    tiktok_profile_url: Optional[str] = Field(default=None)
    tiktok_is_verified: bool = Field(default=False)
    tiktok_privacy_options_json: Optional[str] = Field(default=None)

    google_business_access_token: Optional[str] = Field(default=None)
    google_business_refresh_token: Optional[str] = Field(default=None)
    google_business_token_expires_at: Optional[datetime] = Field(default=None)
    google_business_account_name: Optional[str] = Field(default=None)
    google_business_account_display_name: Optional[str] = Field(default=None)
    google_business_location_name: Optional[str] = Field(default=None)
    google_business_location_title: Optional[str] = Field(default=None)
    google_business_location_store_code: Optional[str] = Field(default=None)
    google_business_location_category: Optional[str] = Field(default=None)
    google_business_locations_json: Optional[str] = Field(default="[]")

    created_at: datetime = Field(default_factory=utcnow)


class Robot(SQLModel, table=True):
    __tablename__ = "robot"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    public_id: str = Field(index=True, unique=True)

    title: str
    description: str = Field(default="")
    avatar_data: Optional[str] = Field(default=None)

    system_instructions: str
    model: str = Field(default="gpt-4o-mini")

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

    knowledge_text: str = Field(default="")
    knowledge_files_json: str = Field(default="[]")
    skybob: str = Field(default="")

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
    user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
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
    user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    client_id: str = Field(index=True)
    agent_key: str = Field(index=True)
    nucleus_json: str
    output_text: str
    created_at: datetime = Field(default_factory=utcnow, index=True)
class BobarColumn(SQLModel, table=True):
    __tablename__ = "bobar_column"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    name: str = Field(default="Nova coluna")
    position: int = Field(default=0, index=True)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class BobarCard(SQLModel, table=True):
    __tablename__ = "bobar_card"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    column_id: int = Field(foreign_key="bobar_column.id", index=True)

    title: str = Field(default="Novo card")
    card_type: str = Field(default="manual", index=True)
    source_kind: Optional[str] = Field(default=None, index=True)
    source_label: Optional[str] = Field(default=None)

    content_text: str = Field(default="")
    note: str = Field(default="")
    position: int = Field(default=0, index=True)
    structure_json: str = Field(default="{}")

    created_at: datetime = Field(default_factory=utcnow, index=True)
    updated_at: datetime = Field(default_factory=utcnow, index=True)
