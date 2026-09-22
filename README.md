# PitchMaster AI

PitchMaster AI is a **Distributed Football Intelligence System** designed to provide real-time tactical insights and sentiment analysis using multi-agent orchestration and RAG.

## Features

- **Real-time Telemetry Ingestion**: High-concurrency ingestion of player metrics via FastAPI and Redis.
- **Multi-Agent Orchestration**: Specialized agents for Tactical Analysis and Sentiment Analysis using LangChain.
- **RAG Pipeline**: Retrieval-Augmented Generation over Pinecone, with metadata-filtered top-k retrieval. Sourced from a 34K-player Kaggle/Transfermarkt dataset; `scripts/load_datasets.py` ingests a configurable slice (default 200, `--limit` to scale up).
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

## API Usage

- **Health Check**: `GET /health`
- **Ingest Telemetry**: `POST /api/v1/ingest/telemetry`
- **Analyze**: `POST /api/v1/analyze`

## Testing

Run the test suite:
```bash
pytest
```
