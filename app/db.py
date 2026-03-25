from __future__ import annotations

from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy import text

from .config import DATABASE_URL

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)


def _ensure_sqlite_table_columns(table_name: str, wanted: dict[str, str]) -> None:
    with engine.begin() as conn:
        result = conn.execute(text(f"PRAGMA table_info('{table_name}')"))
        rows = result.fetchall()
        if not rows:
            return

        existing = {row[1] for row in rows}
        for column, sql_type in wanted.items():
            if column not in existing:
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column} {sql_type}"))


def _ensure_sqlite_columns() -> None:
    if not DATABASE_URL.startswith("sqlite"):
        return

    _ensure_sqlite_table_columns(
        "user",
        {
            "instagram_meta_access_token": "TEXT",
            "instagram_meta_token_expires_at": "TIMESTAMP",
            "instagram_account_id": "TEXT",
            "instagram_page_id": "TEXT",
            "instagram_username": "TEXT",
            "facebook_user_access_token": "TEXT",
            "facebook_user_token_expires_at": "TIMESTAMP",
            "facebook_page_id": "TEXT",
            "facebook_page_name": "TEXT",
            "facebook_page_username": "TEXT",
            "facebook_page_access_token": "TEXT",
            "facebook_pages_json": "TEXT",
            "youtube_access_token": "TEXT",
            "youtube_refresh_token": "TEXT",
            "youtube_token_expires_at": "TIMESTAMP",
            "youtube_channel_id": "TEXT",
            "youtube_channel_title": "TEXT",
            "youtube_channel_handle": "TEXT",
            "youtube_channel_thumbnail": "TEXT",
            "tiktok_access_token": "TEXT",
            "tiktok_refresh_token": "TEXT",
            "tiktok_token_expires_at": "TIMESTAMP",
            "tiktok_refresh_token_expires_at": "TIMESTAMP",
            "tiktok_open_id": "TEXT",
            "tiktok_scope": "TEXT",
            "tiktok_display_name": "TEXT",
            "tiktok_username": "TEXT",
            "tiktok_avatar_url": "TEXT",
            "tiktok_profile_url": "TEXT",
            "tiktok_is_verified": "BOOLEAN",
            "tiktok_privacy_options_json": "TEXT",
            "google_business_access_token": "TEXT",
            "google_business_refresh_token": "TEXT",
            "google_business_token_expires_at": "TIMESTAMP",
            "google_business_account_name": "TEXT",
            "google_business_account_display_name": "TEXT",
            "google_business_location_name": "TEXT",
            "google_business_location_title": "TEXT",
            "google_business_location_store_code": "TEXT",
            "google_business_location_category": "TEXT",
            "google_business_locations_json": "TEXT",
            "skybob": "TEXT",
        },
    )

    _ensure_sqlite_table_columns(
        "bobar_card",
        {
            "structure_json": "TEXT",
        },
    )


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    _ensure_sqlite_columns()


def get_session():
    with Session(engine) as session:
        yield session
