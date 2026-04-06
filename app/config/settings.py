from __future__ import annotations

import os
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
    agent_name: str = 'TrustTrade AI Agent'
    agent_source: str = 'python-agent'
    default_role: str = 'member'
    max_history: int = 8
    knowledge_init_wait_seconds: float = 8.0
    warmup_wait_seconds: float = float(os.getenv('WARMUP_WAIT_SECONDS', '20'))
    enable_semantic_search: bool = _env_flag('ENABLE_SEMANTIC_SEARCH', True)
    groq_api_key: str = os.getenv('GROQ_API_KEY', '')
    groq_model: str = 'llama-3.3-70b-versatile'
    warmup_api_key: str = os.getenv('AGENT_WARMUP_KEY', '')
    
    # Backend Integration
    backend_api_url: str = os.getenv('BACKEND_API_URL', 'http://localhost:5001')
    agent_internal_key: str = os.getenv('AGENT_INTERNAL_KEY', 'trusttrade-local-agent')
    backend_request_timeout_seconds: float = float(os.getenv('BACKEND_REQUEST_TIMEOUT_SECONDS', '10'))
    
    # MongoDB Configuration
    mongodb_uri: str = os.getenv('MONGODB_URL', os.getenv('MONGODB_URI', 'mongodb://localhost:27017/trusttrade'))
    database_name: str = os.getenv('DATABASE_NAME', 'assetdirect')
    knowledge_collection_name: str = os.getenv('KNOWLEDGE_COLLECTION_NAME', 'knowledges')
    conversation_collection_name: str = os.getenv('CONVERSATION_COLLECTION_NAME', 'conversations')


settings = Settings()
