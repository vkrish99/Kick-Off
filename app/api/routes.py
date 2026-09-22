from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.models.schemas import (
    IngestTelemetryRequest,
    AnalysisRequest,
    AnalysisResponse,
    PlayerProfile,
)
from app.services.cache import redis_service
from app.services.rag import rag_service
from app.agents.orchestrator import orchestrator
from app.core.security import sanitize_query
from app.core.config import get_settings
from typing import List
import logging

settings = get_settings()

logger = logging.getLogger(__name__)
api_router = APIRouter()

# Rate limiter — registered on app.state in main.py
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Telemetry Ingestion
# ---------------------------------------------------------------------------
@api_router.post("/ingest/telemetry", status_code=202, tags=["Ingestion"])
@limiter.limit("60/minute")
async def ingest_telemetry(
    request: Request,
    payload: IngestTelemetryRequest,
    background_tasks: BackgroundTasks,
):
    """
    Ingest real-time player telemetry.
    Items are pushed to the Redis `telemetry_queue` and processed asynchronously
    by the background telemetry worker.
    """
    try:
        for item in payload.data:
            await redis_service.enqueue_task(
                "telemetry_queue", item.model_dump(mode="json")
            )
        return {"status": "queued", "count": len(payload.data)}
    except Exception as exc:
        logger.error("Failed to enqueue telemetry: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")


# ---------------------------------------------------------------------------
# Player Profile Ingestion
# ---------------------------------------------------------------------------
@api_router.post("/ingest/profiles", status_code=201, tags=["Ingestion"])
async def ingest_profiles(profiles: List[PlayerProfile]):
    """
    Ingest player profiles into Pinecone vector store for RAG retrieval.
    """
    try:
        count = await rag_service.ingest_profiles(profiles)
        return {"status": "ingested", "count": count}
    except Exception as exc:
        logger.error("Failed to ingest profiles: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")


# ---------------------------------------------------------------------------
# Queue Status
# ---------------------------------------------------------------------------
@api_router.get("/queue/status", tags=["Observability"])
async def queue_status():
    """
    Returns the current depth of the Redis telemetry queue and number of
    items processed since start-up.
    """
    from app.services.telemetry_worker import get_processed_count

    try:
        await redis_service.connect()
        depth = await redis_service.queue_length("telemetry_queue")
        return {
            "queue": "telemetry_queue",
            "depth": depth,
            "processed_total": get_processed_count(),
        }
    except Exception as exc:
        logger.error("Failed to get queue status: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Queue service temporarily unavailable.")


# ---------------------------------------------------------------------------
# Multi-Agent Analysis
# ---------------------------------------------------------------------------
@api_router.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
@limiter.limit("60/minute")
async def analyze_query(request: Request, payload: AnalysisRequest):
    """
    Route a natural-language query to the most appropriate specialist agent:
    - **TACTICAL** — when `telemetry_context` is supplied and query is tactical
    - **FORMATION** — when `telemetry_context` is supplied and query is about shape
    - **SENTIMENT** — when `news_context` is supplied and query is about morale
    - **RAG** — historical profile lookups via Pinecone
    - **GENERAL** — fallback plain-text response
    """
    # Sanitize user-supplied text fields before passing to LLM chains
    sanitize_query(payload.query)
    if payload.news_context:
        sanitize_query(payload.news_context)

    try:
        telemetry_context = (
            [t.model_dump() for t in payload.telemetry_context]
            if payload.telemetry_context
            else None
        )
        insight, route = await orchestrator.process_query(
            query=payload.query,
            telemetry_context=telemetry_context,
            news_context=payload.news_context,
            filters=payload.filters,
        )
        return AnalysisResponse(insight=insight, route_used=route)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Analysis failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error.")


# ---------------------------------------------------------------------------
# Client Configuration
# ---------------------------------------------------------------------------
@api_router.get("/config", tags=["Observability"])
async def get_client_config() -> dict:
    """
    Exposes non-sensitive runtime constants to the frontend (pitch dimensions,
    thresholds). Keeps the dashboard in sync with backend config without
    hardcoding values in JavaScript.
    """
    return {
        "pitch_length_m": settings.PITCH_LENGTH_M,
        "pitch_width_m": settings.PITCH_WIDTH_M,
        "fatigue_high_threshold": settings.FATIGUE_HIGH_THRESHOLD,
        "max_telemetry_batch": settings.MAX_TELEMETRY_BATCH,
    }
