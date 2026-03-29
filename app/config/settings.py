from __future__ import annotations

import os
import sys
# 🔥 Fix for OpenMP / SHM2 failure on Mac/Conda environments
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Path resolution:
# Agent/app/config/settings.py -> parents[2] reaches the Agent/ directory
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / '.env')


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {'1', 'true', 'yes', 'on'}


@dataclass(frozen=True)
class Settings:
    agent_name: str = os.getenv('AGENT_NAME', 'TrustTrade AI Agent')
    agent_source: str = os.getenv('AGENT_SOURCE', 'python-agent')
    default_role: str = os.getenv('AGENT_DEFAULT_ROLE', 'member')
    max_history: int = int(os.getenv('AGENT_MAX_HISTORY', '8'))
    knowledge_init_wait_seconds: float = float(os.getenv('KNOWLEDGE_INIT_WAIT_SECONDS', '0.35'))
    enable_semantic_search: bool = _env_flag('ENABLE_SEMANTIC_SEARCH', sys.platform != 'darwin')
    groq_api_key: str = os.getenv('GROQ_API_KEY', '')
    groq_model: str = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
    
    # MongoDB Configuration
    mongodb_uri: str = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/trusttrade')
    database_name: str = os.getenv('DATABASE_NAME', 'trusttrade')
    knowledge_collection_name: str = os.getenv('KNOWLEDGE_COLLECTION_NAME', 'knowledges')


settings = Settings()
