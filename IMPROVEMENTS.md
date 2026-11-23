# Kick-Off (PitchMaster AI) — Improvement Plan

This plan comes from a full read of the codebase (Sept 2026). Work through it in order: each phase
depends on the one before it. Every task names the files to change and a **Done when** check.

**The core problem:** the README describes a *distributed, real-time* system, but the telemetry
pipeline and the AI analysis are not connected. Telemetry goes into Redis. The worker pops it,
logs it, optionally writes it to InfluxDB, and throws it away. `/analyze` never reads any of that
data; it only sees `telemetry_context` that the *client* sends in the request body. Phase 1
connects them, and it's the change that makes the project's main claim true.

---

## Phase 0 — Correctness and security fixes

**Done (Nov 2026):** 0.1, 0.2 (RAG kept in the orchestrator with a cached chain rather than a
separate `scout.py`), 0.3 (except the `LLM_LATENCY` histogram), 0.4, and 0.6 (API key on
`/ingest/*`, CORS, rate limit on `/ingest/profiles`, delimited RAG prompt, `utcnow`). Tests are in
[`tests/test_hardening.py`](tests/test_hardening.py). **Still open:** 0.5 (typed `filters`), 0.7
(batched `RPUSH`), and delimiting user text in the tactical/formation/sentiment prompts.

### 0.1 Blocking calls inside async handlers
[`rag.py`](app/services/rag.py) and [`vector_store.py`](app/services/vector_store.py) run
SentenceTransformer `encode()` (CPU-bound) and the synchronous Pinecone SDK inside
`async def` routes. Every RAG query or ingest **blocks the event loop**, which stalls telemetry
ingestion and all other requests.
**Do:** Wrap `retriever.invoke`, `add_documents`, and `initialize_index` in `await asyncio.to_thread(...)`.
Call `initialize_index()` **once** at startup in the lifespan, not on every `/ingest/profiles` call.
**Done when:** A test runs a slow fake embedder and a `/health` request at the same time, and
`/health` returns in under 50 ms.

### 0.2 RAG route ignores missing config and missing results
In [`orchestrator.py`](app/agents/orchestrator.py), the RAG branch:
- builds `ChatGroq` with no `GROQ_API_KEY` check and creates a new client on every request.
- still calls the LLM when retrieval returns **zero documents**, so the model invents players.
- returns no sources.

**Do:** Move RAG synthesis into its own agent (`app/agents/scout.py`) with the same lazy-chain,
retry, and timeout pattern as the other agents. If `docs` is empty, return "No matching players
in the index" without calling the LLM. Add `sources: list[{player_id, name, score}]` to
`AnalysisResponse`. The system prompt must tell the model to answer **only** from the retrieved
context.

### 0.3 Router parses the model's reply by substring
`if "TACTICAL" in route` accepts any reply that contains the word. The router is also an extra
LLM call on every request.
**Do:** Put the tokens in an `Enum` and match the reply exactly after `strip().upper()`. On a
parse failure, fall back to GENERAL and log it. Skip the LLM when the request is unambiguous:
`news_context` only means SENTIMENT, and `filters` set means RAG. Record the chosen route and
latency in the `LLM_LATENCY` histogram, which is defined in [`metrics.py`](app/core/metrics.py)
but never used.

### 0.4 Pinecone vector IDs collide
[`vector_store.add_documents`](app/services/vector_store.py) falls back to `f"doc-{i}"`, which
restarts at 0 on every call, so later batches overwrite earlier ones.
**Do:** Make `player_id` required (it already is in `PlayerProfile`), and remove the fallback or
use `uuid4()`.

### 0.5 `filters` goes to Pinecone unchecked
`AnalysisRequest.filters: Optional[dict]` is passed through as-is.
**Do:** Replace it with a typed model `RagFilters(position: str | None, team: str | None)` and
build the Pinecone filter (`{"position": {"$eq": ...}}`) on the server.

### 0.6 API hardening
- **CORS:** [`main.py`](app/main.py) sets `allow_origins=["*"]` together with
  `allow_credentials=True`. That combination is invalid under the spec and unsafe. Use
  `allow_credentials=False` when the origin is `*`.
- **Auth:** `/ingest/*` has no authentication, so anyone can write to Pinecone. Add an `X-API-Key`
  dependency, set by `API_KEY` in `.env`, on every write endpoint.
- **Rate limit:** `/ingest/profiles` has no `@limiter.limit`. Add one.
- **Prompt-injection regex:** [`security.py`](app/core/security.py) blocks a handful of phrases,
  which is easy to bypass. Keep it as a cheap filter, but make the defense structural: put user
  text in clearly delimited sections of the prompt, and instruct the model to treat that section
  as data. Rename the module docstring so it no longer claims the regex "protects" against
  injection. Remove `MAX_INPUT_LENGTH` there and use `settings.MAX_QUERY_LENGTH` instead.
- Replace `datetime.utcnow` (deprecated) with `datetime.now(timezone.utc)` in [`schemas.py`](app/models/schemas.py).

### 0.7 Ingest enqueues one item at a time
[`routes.ingest_telemetry`](app/api/routes.py) awaits one `RPUSH` per item.
**Do:** Send the batch with a single `RPUSH queue v1 v2 ...`, or a pipeline.

---

## Phase 1 — Connect the pipeline

### 1.1 Replace the Redis list with Redis Streams
`BLPOP` gives at-most-once delivery: if the worker crashes after popping an item, that item is
lost.
**Do:** In [`cache.py`](app/services/cache.py) and [`telemetry_worker.py`](app/services/telemetry_worker.py),
switch to `XADD telemetry * ...`, a consumer group `workers`, `XREADGROUP` with `COUNT 100 BLOCK 1000`,
and `XACK` after processing. On startup, reclaim stuck messages with `XAUTOCLAIM`. Keep the
in-memory fallback for dev.
**Done when:** A test kills the worker mid-batch, restarts it, and every message is processed exactly once.

### 1.2 Keep live match state
**Do:** The worker writes each player's latest reading to a Redis hash `player:{id}:latest`,
with a 10-minute TTL, and adds the player to a set `match:{match_id}:players`. Add
`match_id: str` to `TelemetryData`.

### 1.3 `/analyze` reads server-side state
**Do:** When the request has no `telemetry_context` but has a `match_id`, load the latest state
of every player from Redis. The dashboard's "simulate" button then posts to `/ingest/telemetry`,
and analysis runs on what the pipeline actually processed. That's the distributed flow the README
describes.

### 1.4 Ground the LLM with computed stats
Today the agents receive `str(telemetry_data)`, a Python repr of dicts that includes datetime
objects.
**Do:** Add `app/services/match_stats.py`, which computes the numbers in plain Python before the
LLM call:
- fatigue ranking and players above `FATIGUE_HIGH_THRESHOLD`.
- average position per player, team shape (defensive line height, width, compactness), and the
  largest gaps between lines.
- speed and heart-rate outliers.

Send the LLM a compact table plus these computed facts, and tell it to cite them. Return the facts
in `AnalysisResponse.related_metrics`. That field already exists and is never filled.

### 1.5 Make InfluxDB writes efficient
[`_write_to_influxdb`](app/services/telemetry_worker.py) opens a **new client connection for
every item**.
**Do:** Create one `InfluxDBClientAsync` in the lifespan and write in batches of up to 100
points, flushing at least every second. Tag each point with `match_id`. Use the item's own
`timestamp` instead of write time.

### 1.6 Metrics that mean something
- Set `QUEUE_DEPTH` from `XLEN`, or better, consumer-group lag.
- Add `pitchmaster_worker_processed_total`, `pitchmaster_worker_errors_total`, and an end-to-end
  latency histogram from ingest to processed. Delete the module-global `_processed_count`, which
  is wrong with more than one worker process.
- Use the route template (`request.scope["route"].path`) for the `endpoint` label instead of the
  raw URL.
- Add Prometheus and Grafana to `docker-compose.yml` with a provisioned dashboard JSON.

### 1.7 Scale out
With consumer groups in place, run the worker as its own compose service
(`python -m app.services.telemetry_worker`). The API starts it in-process only when
`RUN_WORKER=true`. Then `docker compose up --scale worker=3` shows real horizontal scaling.
Document it in the README with a load test (1.8).

### 1.8 Load test
Add `scripts/loadtest.py` using `locust`. POST telemetry batches and report throughput, p95
ingest latency, and queue lag with 1 and 3 workers. Put the actual numbers in the README; they
make a stronger resume line than "high-concurrency".

---

## Phase 2 — RAG quality

1. **Richer documents:** [`rag.ingest_profiles`](app/services/rag.py) embeds only
   name, position, team, skills, and summary. `load_datasets.py` already computes goals,
   minutes, market value, and injuries, so put them into metadata as numeric fields. That
   allows filters like `market_value >= 50M` and `age <= 23`.
2. **Hybrid search:** combine dense retrieval with a sparse/BM25 vector (Pinecone supports sparse
   values) so exact names like "Ødegaard" still match.
3. **A real eval set:** [`eval_retrieval.py`](scripts/eval_retrieval.py) only tests
   self-retrieval ("<name> <team>"), which almost always hits. Add
   `scripts/eval_queries.jsonl` with 30–50 hand-written questions and their expected
   player IDs, for example "young left-footed centre-back under 25 at a Bundesliga club".
   Report Hit@k and MRR for dense and hybrid search, and put the table in the README.
4. **Answer eval:** for each eval question, check whether every player named in the answer
   appears in the retrieved sources, and report the hallucination rate.

---

## Phase 3 — Testing and CI

- ~~Add CI~~ **Done:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs `ruff check`
  (real-error rules only, see [`ruff.toml`](ruff.toml)) and `pytest` on Python 3.12 with CPU-only
  torch. Still to add: a `docker build` job (after the CPU-torch Dockerfile fix in Phase 4) and a
  Redis service container once Phase 1.1 needs a real Redis to test consumer groups.
- Mock Groq and Pinecone in unit tests, as `test_agents.py` already does. Add tests for 0.2
  (empty retrieval returns no LLM call), 0.3 (enum parsing), 0.4 (unique IDs), 0.6 (401 without
  an API key), and 1.1 (redelivery after a crash).
- ~~Pin dependency versions~~ **Done:** `requirements.in` → pinned `requirements.txt`. Was: `requirements.txt` used only `>=`, so a LangChain release can break the
  build. Generate a lock file with `pip-compile` or `uv lock`.

---

## Phase 4 — Docker and developer experience

- ~~**Image size**~~ **Done:** replaced `sentence-transformers` with `fastembed` (no torch). Was: `sentence-transformers` installs the **CUDA** build of torch, which is several
  GB. In the Dockerfile, install CPU torch first:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- ~~**Model download**~~ **Done:** the Dockerfile downloads it at build time. Was: the embedding model downloads on the first request. Download it at build
  time with `RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`
  and set `HF_HOME` to a directory `appuser` can read.
- Add `healthcheck:` entries to compose, and use `depends_on: condition: service_healthy`.
- Move the InfluxDB password and token out of `docker-compose.yml` into `.env`.
- Add a `Makefile` or `justfile` with `up`, `seed`, `eval`, `test`, and `load` targets.
- **Datasets:** `kaggle_football/` (~620 MB) and `football-datasets-main/` (~120 MB) are
  gitignored, but nothing explains how to get them. Add `scripts/download_datasets.sh`
  (Kaggle CLI plus a git clone of the Transfermarkt repo) and name both sources with their
  licences in the README.

---

## Phase 5 — README rewrite

Rewrite [`README.md`](README.md) after Phases 0–1 so every claim is backed by the code:
- One name. The repo is **Kick-Off** and the app calls itself **PitchMaster AI**. Pick one, or
  write "Kick-Off (PitchMaster AI)".
- An architecture diagram (Mermaid) showing: dashboard → FastAPI → Redis Stream → N workers →
  Redis state + InfluxDB → `/analyze` → router → agents → Groq; plus Pinecone for RAG.
- Describe it as an **LLM router over specialist agents**, not "multi-agent orchestration",
  unless Phase 6.1 is done.
- Real numbers: load-test throughput and p95 (1.8), and retrieval Hit@k and MRR (Phase 2).
- A full API reference with request examples for `/ingest/telemetry`, `/ingest/profiles`,
  `/analyze`, `/queue/status`, `/config`, and `/metrics`.
- Quickstart: `cp .env.example .env` → `make up` → `make seed` → open `http://localhost:8000`.

---

## Phase 6 — Optional stretch goals

0. ~~**Player availability sync**~~ **Done:** `scripts/sync_player_status.py` pulls API-Football
   injuries, matches players to dataset ids, and stores statuses in Redis and on the Pinecone
   metadata. Next steps: pass unavailable players to the Formation agent (item 1 below), filter
   RAG with `{"status": {"$ne": "out"}}` when the question is about picking a lineup, and run the
   sync on a schedule inside docker-compose.

1. **Real multi-agent flow:** replace the one-shot router with a LangGraph graph where, for
   example, the Formation agent can call the Scout (RAG) agent to find a replacement for a
   fatigued player. "Sub out X for Y from the bench, and here's why" is a much stronger demo.
2. **Streaming answers:** stream `/analyze` over SSE so the dashboard shows tokens as they
   arrive.
3. **Live updates:** push worker alerts (high fatigue) to the dashboard over WebSocket.
4. **Replay a real match:** feed `game_events.csv` from the Kaggle data through the pipeline
   minute by minute to demo the system on real events instead of random telemetry.

---

## Suggested order and scope

| Order | Items | Effort |
| --- | --- | --- |
| 1 | Phase 0 | ~1 day |
| 2 | Phase 3 (CI, before larger changes) | ~0.5 day |
| 3 | 1.1 – 1.4 (connect the pipeline) | ~2 days |
| 4 | 1.5 – 1.8 (metrics, scaling, load test) | ~1–2 days |
| 5 | Phase 2 (eval set + hybrid) | ~1 day |
| 6 | Phase 4 + Phase 5 (docker, README) | ~1 day |
| 7 | Phase 6 | as time allows |

**Resume line once Phases 0–3 are done:** "Built a distributed football analytics pipeline:
FastAPI ingestion into Redis Streams with horizontally scaled consumer-group workers (N msgs/s,
p95 X ms), and live match state feeding an LLM-routed set of tactical, formation, and scouting
agents. Hybrid RAG over 34K players reaches Hit@3 of Y on a hand-labeled eval set."
