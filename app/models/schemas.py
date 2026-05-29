from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Optional
from datetime import datetime

# ---------------------------------------------------------------------------
# Telemetry Data Model
# ---------------------------------------------------------------------------
class TelemetryData(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "player_id": "haaland_9",
            "speed": 28.4,
            "distance_covered": 9.2,
            "heart_rate": 172,
            "position_x": 85.0,
            "position_y": 34.0,
            "fatigue_index": 0.62,
        }
    })

    player_id: str = Field(..., min_length=1, max_length=64)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    speed: float = Field(..., ge=0.0, le=120.0)             # km/h (physiological cap)
    distance_covered: float = Field(..., ge=0.0, le=20.0)   # km per session
    heart_rate: int = Field(..., ge=30, le=250)              # bpm
    position_x: float = Field(..., ge=0.0, le=105.0)        # metres (pitch length)
    position_y: float = Field(..., ge=0.0, le=68.0)         # metres (pitch width)
    fatigue_index: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)  # 0=fresh, 1=exhausted

# ---------------------------------------------------------------------------
# Live Match Event
# ---------------------------------------------------------------------------
class MatchEvent(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "event_type": "goal",
            "player_id": "haaland_9",
            "minute": 67,
            "description": "Header from corner kick.",
        }
    })

    event_type: str           # e.g. "goal", "foul", "substitution", "card"
    player_id: str
    minute: int = Field(..., ge=0, le=120)  # 0-90 regular + extra time
    description: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# ---------------------------------------------------------------------------
# News / Sentiment Data
# ---------------------------------------------------------------------------
class NewsSentimentData(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "player_id": "debruyne_17",
            "source": "The Athletic",
            "content": "Kevin De Bruyne returned to full training today ahead of the derby.",
            "sentiment_score": 0.8,
            "published_at": "2025-04-15T10:30:00Z",
        }
    })

    player_id: str
    source: str
    content: str
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)  # -1=negative, 1=positive
    published_at: datetime

# ---------------------------------------------------------------------------
# Player Profile  (stored in Vector DB)
# ---------------------------------------------------------------------------
class PlayerProfile(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "player_id": "haaland_9",
            "name": "Erling Haaland",
            "position": "Striker",
            "team": "Manchester City",
            "skills": ["finishing", "pace", "aerial ability", "hold-up play"],
            "recent_performance_summary": "Scored 2 goals vs Arsenal. Averaging 0.95 xG per 90.",
        }
    })

    player_id: str
    name: str
    position: str
    team: str
    skills: List[str]
    recent_performance_summary: Optional[str] = None

# ---------------------------------------------------------------------------
# API Request / Response Models
# ---------------------------------------------------------------------------
class IngestTelemetryRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "data": [
                {
                    "player_id": "haaland_9",
                    "speed": 28.4,
                    "distance_covered": 9.2,
                    "heart_rate": 172,
                    "position_x": 85.0,
                    "position_y": 34.0,
                    "fatigue_index": 0.62,
                }
            ]
        }
    })

    data: List[TelemetryData]

    @field_validator("data")
    @classmethod
    def check_batch(cls, v: List[TelemetryData]) -> List[TelemetryData]:
        from app.core.config import get_settings
        s = get_settings()
        if not v:
            raise ValueError("Telemetry batch must not be empty")
        if len(v) > s.MAX_TELEMETRY_BATCH:
            raise ValueError(
                f"Batch too large: {len(v)} items (max {s.MAX_TELEMETRY_BATCH})"
            )
        return v


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "Which players are showing signs of fatigue and should be substituted?",
            "telemetry_context": [
                {
                    "player_id": "haaland_9",
                    "speed": 18.1,
                    "distance_covered": 11.3,
                    "heart_rate": 188,
                    "position_x": 72.0,
                    "position_y": 38.0,
                    "fatigue_index": 0.88,
                }
            ],
        }
    })

    query: str = Field(..., min_length=1, max_length=2000)
    filters: Optional[dict] = None
    # Optional rich context callers can pass to trigger specialist agents
    telemetry_context: Optional[List[TelemetryData]] = None
    news_context: Optional[str] = None

class AnalysisResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "insight": "Haaland's fatigue index of 0.88 is above the critical threshold. Recommend substitution before the 80th minute.",
            "route_used": "TACTICAL",
            "related_metrics": None,
        }
    })

    insight: str
    related_metrics: Optional[dict] = None
    route_used: Optional[str] = None   # surfaces which agent handled the query
