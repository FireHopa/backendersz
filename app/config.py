from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1").strip()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data.db").strip()

# SERPER (Google Search via API) — recomendado para "procurar concorrentes"
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
SERPER_GL = os.getenv("SERPER_GL", "br").strip()          # country code
SERPER_HL = os.getenv("SERPER_HL", "pt-br").strip()       # language
SERPER_LOCATION = os.getenv("SERPER_LOCATION", "Brazil").strip()


# Web Search (Serper) para chat
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").strip().lower() in {"1","true","yes","y"}
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
