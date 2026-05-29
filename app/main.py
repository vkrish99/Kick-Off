"""
PitchMaster AI — Application Entry Point
"""
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.api import routes
from app.api.routes import limiter
from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.core.metrics import REQUEST_COUNT
from app.services.telemetry_worker import telemetry_worker

settings = get_settings()

# Configure logging (JSON in production, human-readable in dev)
configure_logging(json_logs=settings.LOG_JSON)

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — replaces deprecated @app.on_event("startup/shutdown")
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up %s v%s", settings.PROJECT_NAME, settings.VERSION)

    # Warn loudly about missing API keys so misconfiguration is obvious at boot
    if not settings.GROQ_API_KEY:
        logger.warning(
            "GROQ_API_KEY is not set — AI agent endpoints (/analyze) will return 503 "
            "until a valid key is configured in .env"
        )
    if not settings.PINECONE_API_KEY:
        logger.warning(
            "PINECONE_API_KEY is not set — RAG profile ingestion/retrieval will return 503 "
            "until a valid key is configured in .env"
        )

    worker_task = None
    try:
        worker_task = asyncio.create_task(telemetry_worker(), name="telemetry_worker")
        logger.info("Telemetry worker task started.")
    except Exception as exc:
        logger.warning("Telemetry worker failed to start (Redis unavailable?): %s", exc)

    yield  # Application is running

    logger.info("Shutting down %s", settings.PROJECT_NAME)
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "**PitchMaster AI** — Distributed Football Intelligence System.\n\n"
        "Real-time telemetry ingestion, multi-agent tactical & sentiment analysis, "
        "and RAG-powered player profile retrieval."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
cors_origins = (
    ["*"]
    if settings.CORS_ORIGINS.strip() == "*"
    else [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Middleware — Request ID + Latency logging + Prometheus counter
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_id_middleware(request: Request, call_next) -> Response:
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()
    response: Response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "req_id=%s method=%s path=%s status=%s latency_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code),
    ).inc()
    return response


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(routes.api_router, prefix=settings.API_V1_STR)

# ---------------------------------------------------------------------------
# Static Dashboard UI
# ---------------------------------------------------------------------------
static_dir = Path("app/static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", tags=["Dashboard"], response_class=HTMLResponse)
async def dashboard():
    """Serves the main interactive dashboard UI."""
    html_path = static_dir / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Dashboard is building... Please wait.</h1>", status_code=404)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------------
@app.get("/metrics", tags=["Observability"], include_in_schema=False)
async def metrics():
    """Exposes Prometheus metrics for scraping."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
    }
