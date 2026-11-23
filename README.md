# PitchMaster AI

PitchMaster AI is a **Distributed Football Intelligence System** designed to provide real-time tactical insights and sentiment analysis using multi-agent orchestration and RAG.

## Features

- **Real-time Telemetry Ingestion**: High-concurrency ingestion of player metrics via FastAPI and Redis.
- **Multi-Agent Orchestration**: Specialized agents for Tactical Analysis and Sentiment Analysis using LangChain.
- **RAG Pipeline**: Retrieval-Augmented Generation over Pinecone, with metadata-filtered top-k retrieval. Sourced from a 34K-player Kaggle/Transfermarkt dataset; `scripts/load_datasets.py` ingests a configurable slice (default 200, `--limit` to scale up).
- **Player Availability Sync**: Pulls current injuries and doubts from [API-Football](https://www.api-football.com), maps them onto dataset player ids, and exposes them to the API and the RAG agent.
- **Distributed Architecture**: Built with Python, FastAPI, Redis, and scalability in mind.

## Architecture

- **API Layer**: FastAPI
- **Queue/Cache**: Redis
- **Vector Database**: Pinecone
- **LLM/Agents**: Groq (Llama 3.3 70B), LangChain

## Setup

1. **Environment Variables**:
   Copy `.env.example` to `.env` and fill in your API keys (Groq, Pinecone).
   ```bash
   cp .env.example .env
   ```
   Set `API_KEY` too before exposing the server publicly: the `/ingest/*` write endpoints then
   require it in an `X-API-Key` header. The seed scripts read it from `.env`, and
   `PITCHMASTER_URL` points them at a deployed server instead of `localhost`.

2. **Run everything with Docker Compose** (app + Redis + InfluxDB):
   ```bash
   docker-compose up --build
   ```
   The API is then live at `http://localhost:8000`.

   Or run the app locally instead, against Dockerized infrastructure:
   ```bash
   docker-compose up -d redis influxdb
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

   `requirements.txt` pins every version for reproducible builds; edit `requirements.in` to
   change dependencies. Embeddings use `fastembed` (ONNX Runtime, no torch), which produces the
   same vectors as the `all-MiniLM-L6-v2` sentence-transformers model. The Docker image bakes the
   model in, and the container used about 320 MB of RAM with it loaded, so a 1 GB server is enough.

## API Usage

- **Health Check**: `GET /health`
- **Ingest Telemetry**: `POST /api/v1/ingest/telemetry` (needs `X-API-Key` when `API_KEY` is set)
- **Ingest Profiles**: `POST /api/v1/ingest/profiles` (needs `X-API-Key` when `API_KEY` is set)
- **Analyze**: `POST /api/v1/analyze`. Profile answers (route `RAG`) include the `sources` they
  were built from; if no profile matches, the route is `RAG_NO_RESULTS` and the LLM isn't called.
- **Player status**: `GET /api/v1/players/{player_id}/status` → `out`, `doubtful`, `available`, or `unknown` (no sync yet)
- **Unavailable players**: `GET /api/v1/players/unavailable?team=Arsenal`

## Player Availability Sync

Real injury data comes from API-Football's official API (FotMob and similar sites disallow automated access to their data endpoints, so they are not scraped).

1. Get a free key at [dashboard.api-football.com](https://dashboard.api-football.com) (100 requests/day) and set `API_FOOTBALL_KEY` in `.env`.
2. Start Redis: `docker-compose up -d redis`
3. Preview what would be synced: `python scripts/sync_player_status.py --dry-run`
4. Sync: `python scripts/sync_player_status.py --leagues 39,140 --season 2025`

Each run uses one request per league, and responses are cached for 6 hours. The script:
- keeps each player's most recent injury row, and ignores rows more than 10 days old;
- matches API-Football players to dataset `player_id`s by club and name (e.g. "E. Haaland" at "Manchester City"), and only accepts a match when exactly one player fits. Confirmed matches are saved to `data/player_id_map.json`, and you can edit that file to fix a mapping. Unmatched players are listed in the output (usually players who changed clubs after the dataset snapshot);
- stores statuses in Redis and sets `status` / `status_reason` metadata on the Pinecone vectors, so RAG answers mention who is unavailable. Players who were flagged by the previous sync but not this one are reset to `available`.

Schedule it once a day (cron or Windows Task Scheduler) to keep statuses current; they expire after 36 hours if the sync stops running.

## Testing

Run the test suite:
```bash
pytest
```
