from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", case_sensitive=True)

    PROJECT_NAME: str = "PitchMaster AI"
    VERSION: str = "0.3.0"
    API_V1_STR: str = "/api/v1"

    # Groq — free LLM inference
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_FAST_MODEL: str = "llama-3.1-8b-instant"

    # Pinecone — vector database
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENV: Optional[str] = None
    PINECONE_INDEX_NAME: str = "pitchmaster-index"

    # Legacy — kept for backwards compatibility but not required
    OPENAI_API_KEY: Optional[str] = None

    # Infrastructure
    REDIS_URL: str = "redis://localhost:6379"

    # CORS — comma-separated list of allowed origins (default: allow all)
    CORS_ORIGINS: str = "*"

    # API key required (as the X-API-Key header) on write endpoints (/ingest/*).
    # Leave unset only for local development; always set it on a public deployment.
    API_KEY: Optional[str] = None

    # Structured logging — set to "true" to emit JSON logs (recommended for production)
    LOG_JSON: bool = False

    # InfluxDB — optional time-series persistence for processed telemetry
    INFLUXDB_URL: Optional[str] = None
    INFLUXDB_TOKEN: Optional[str] = None
    INFLUXDB_ORG: str = "pitchmaster"
    INFLUXDB_BUCKET: str = "telemetry"

    # ---------------------------------------------------------------------------
    # Named constants — replaces magic numbers scattered across the codebase
    # ---------------------------------------------------------------------------

    # Telemetry processing thresholds
    FATIGUE_HIGH_THRESHOLD: float = 0.75   # triggers substitution warning
    WORKER_POLL_INTERVAL: float = 0.1      # seconds to sleep when queue is empty
    WORKER_ERROR_BACKOFF: float = 1.0      # seconds to back off after a worker error

    # RAG / Pinecone
    RAG_RETRIEVAL_K: int = 3               # top-k documents returned per query
    PINECONE_BATCH_SIZE: int = 100         # vectors per upsert batch
    PINECONE_CLOUD: str = "aws"
    PINECONE_REGION: str = "us-east-1"
    PINECONE_METRIC: str = "cosine"

    # Embedding model (fastembed/ONNX, runs locally without torch). Produces the same
    # vectors as the sentence-transformers model of the same name.
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384               # must match EMBEDDING_MODEL_NAME

    # API security
    MAX_QUERY_LENGTH: int = 2000           # characters — mirrors security.py
    MAX_TELEMETRY_BATCH: int = 50          # items per /ingest/telemetry request

    # API-Football (api-sports.io) — player injury/availability sync
    API_FOOTBALL_KEY: Optional[str] = None
    API_FOOTBALL_BASE_URL: str = "https://v3.football.api-sports.io"
    API_FOOTBALL_LEAGUES: str = "39"        # comma-separated league ids (39 = Premier League)
    API_FOOTBALL_SEASON: int = 2025         # season start year; free plans may restrict this
    API_FOOTBALL_CACHE_TTL: int = 6 * 3600  # seconds a raw API response is reused
    PLAYER_STATUS_TTL: int = 36 * 3600      # seconds a synced status stays valid
    PLAYER_STATUS_STALE_DAYS: int = 10      # injury rows older than this are ignored

    # Pitch dimensions — single source of truth (metres)
    PITCH_LENGTH_M: float = 105.0
    PITCH_WIDTH_M: float = 68.0


@lru_cache()
def get_settings() -> Settings:
    return Settings()
