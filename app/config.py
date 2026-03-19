from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4").strip()
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1").strip()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data.db").strip()

# SERPER (Google Search via API) — recomendado para "procurar concorrentes"
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
SERPER_GL = os.getenv("SERPER_GL", "br").strip()
SERPER_HL = os.getenv("SERPER_HL", "pt-br").strip()
SERPER_LOCATION = os.getenv("SERPER_LOCATION", "Brazil").strip()

# Web Search (Serper) para chat
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").strip().lower() in {"1", "true", "yes", "y"}
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))

# LINKEDIN OAUTH2
LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID", "").strip()
LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET", "").strip()
LINKEDIN_REDIRECT_URI = os.getenv("LINKEDIN_REDIRECT_URI", "http://localhost:5173/auth/linkedin/callback").strip()

# META / INSTAGRAM
INSTAGRAM_META_APP_ID = os.getenv("INSTAGRAM_META_APP_ID", "").strip()
INSTAGRAM_META_APP_SECRET = os.getenv("INSTAGRAM_META_APP_SECRET", "").strip()
INSTAGRAM_META_REDIRECT_URI = os.getenv("INSTAGRAM_META_REDIRECT_URI", "http://localhost:5173/auth/facebook/callback").strip()

# META / FACEBOOK
FACEBOOK_META_APP_ID = os.getenv("FACEBOOK_META_APP_ID", "").strip()
FACEBOOK_META_APP_SECRET = os.getenv("FACEBOOK_META_APP_SECRET", "").strip()
FACEBOOK_META_REDIRECT_URI = os.getenv("FACEBOOK_META_REDIRECT_URI", "http://localhost:5173/auth/facebook/callback").strip()

META_GRAPH_VERSION = os.getenv("META_GRAPH_VERSION", "v23.0").strip()

# YOUTUBE
YOUTUBE_CLIENT_ID = os.getenv("YOUTUBE_CLIENT_ID", "").strip()
YOUTUBE_CLIENT_SECRET = os.getenv("YOUTUBE_CLIENT_SECRET", "").strip()
YOUTUBE_REDIRECT_URI = os.getenv("YOUTUBE_REDIRECT_URI", "http://localhost:5173/auth/youtube/callback").strip()


# TIKTOK
TIKTOK_CLIENT_KEY = os.getenv("TIKTOK_CLIENT_KEY", "").strip()
TIKTOK_CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET", "").strip()
TIKTOK_REDIRECT_URI = os.getenv("TIKTOK_REDIRECT_URI", "http://localhost:5173/auth/tiktok/callback").strip()

# GOOGLE BUSINESS PROFILE
GOOGLE_BUSINESS_CLIENT_ID = os.getenv("GOOGLE_BUSINESS_CLIENT_ID", "").strip()
GOOGLE_BUSINESS_CLIENT_SECRET = os.getenv("GOOGLE_BUSINESS_CLIENT_SECRET", "").strip()
GOOGLE_BUSINESS_REDIRECT_URI = os.getenv("GOOGLE_BUSINESS_REDIRECT_URI", "http://localhost:5173/auth/google-business/callback").strip()
