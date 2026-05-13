# TrialSight Intelligence

Clinical Trial Evidence Assistant built as a multi-user RAG system for reviewing trial PDFs, retrieving supporting evidence, and generating citation-grounded answers under strict cost and abuse controls.

This is not a generic "chat with PDF" demo. It is a scoped product-style system designed around a real workflow:

- upload trial documents
- isolate data per user
- retrieve relevant evidence
- answer with citations
- enforce hard rate limits before LLM spend happens

## Why It Stands Out

- Niche use case: clinical trial evidence review
- Multi-user architecture with JWT auth and per-user data isolation
- Hybrid retrieval instead of plain keyword search
- Cost-aware by design with Groq as the default LLM provider
- Strict demo-safe rate limiting to protect against abuse and accidental spend
- Lightweight deployment path without heavy local model infrastructure

## Core Capabilities

- PDF upload and ingestion
- Chunked document processing with source-aware citations
- Hybrid retrieval:
  - hashed dense retrieval
  - BM25 keyword search
  - reranking
- Query rewriting before retrieval
- Citation-grounded answer generation
- Streaming responses
- Query history and evaluation logging
- Cache-aware repeated question handling

## Safety and Demo Controls

- JWT authentication
- Per-user document isolation
- Upload size limits
- Upload/day limits
- Query/minute and query/day limits
- Global demo query caps
- Groq request, token, and concurrency guards
- Redis-backed rate limiting with in-memory fallback

## Tech Stack

- FastAPI
- SQLite
- Redis (optional)
- Groq
- BM25 + lightweight dense retrieval
- Static frontend
- Docker

## Project Structure

```text
backend/
  app/
    api/
    core/
    models/
    rag/
    rate_limit/
    schemas/
    services/
frontend/
docker-compose.yml
render.yaml
```

## Local Run

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env and add your GROQ_API_KEY
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
py -3 -m http.server 3000
```

Open:

```text
http://localhost:3000/login.html
```

### Docker (full stack)

```bash
# Copy and configure environment
copy backend\.env.example backend\.env
# Edit backend\.env and add your GROQ_API_KEY

docker compose up --build
```

This starts all three services:
- **Backend** at `http://localhost:8000`
- **Redis** for rate limiting
- **Frontend** at `http://localhost:3000`

## Environment

Copy `backend/.env.example` to `backend/.env`.

Required values:

| Variable | Description |
|----------|-------------|
| `JWT_SECRET` | **Must change for production.** Auto-generated if deployed via `render.yaml`. |
| `GROQ_API_KEY` | Your Groq API key for LLM inference. Get one at [console.groq.com](https://console.groq.com). |

Optional values:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./data/app.db` | SQLite path. Render uses a persistent disk at `/app/data`. |
| `REDIS_URL` | *(unset)* | Redis connection string. If unset, in-memory rate limiting is used. |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model to use. |
| `FRONTEND_ORIGINS` | `http://localhost:3000,...` | Comma-separated CORS origins. Add your deployed frontend URL. |

No API keys are hardcoded. `.env` is ignored by git.

## Deployment

### Backend → Render

**Option A: One-click deploy (recommended)**

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Ari-Han-t/TrialSight-Intelligence)

This uses the `render.yaml` blueprint which auto-configures:
- Docker-based web service on the free tier
- Auto-generated `JWT_SECRET`
- 1 GB persistent disk for SQLite data
- Health check at `/health`

After deploy, set `GROQ_API_KEY` in the Render dashboard under Environment.

**Option B: Manual setup**

1. Create a new **Web Service** on [render.com](https://render.com)
2. Connect your GitHub repository
3. Set:
   - **Root Directory:** `backend`
   - **Runtime:** Docker
   - **Health Check Path:** `/health`
4. Add environment variables:
   - `APP_ENV=production`
   - `JWT_SECRET=<generate a strong random string>`
   - `GROQ_API_KEY=<your key>`
   - `FRONTEND_ORIGINS=https://<your-vercel-app>.vercel.app`
5. Add a **Disk** mounted at `/app/data` (1 GB)
6. Deploy

Your backend will be live at `https://<service-name>.onrender.com`.

### Frontend → Vercel

1. Go to [vercel.com/new](https://vercel.com/new), import this repository
2. Set **Root Directory** to `frontend`
3. Set **Framework Preset** to `Other`
4. Deploy

After deploying:

1. Open `frontend/config.js` and set `apiBase` to your Render backend URL:
   ```js
   window.TRIALSIGHT_CONFIG = {
     apiBase: "https://<your-render-service>.onrender.com",
   };
   ```
2. Commit and push — Vercel will auto-redeploy.
3. Add your Vercel URL to the `FRONTEND_ORIGINS` env var on Render.

### Post-deployment checklist

- [ ] `GROQ_API_KEY` is set on Render
- [ ] `JWT_SECRET` is not the default value
- [ ] `FRONTEND_ORIGINS` includes your Vercel URL
- [ ] `config.js` `apiBase` points to your Render URL
- [ ] Health check passes: `curl https://<service>.onrender.com/health`
- [ ] Sign up, upload a PDF, and run a query end-to-end

## Interview Pitch

TrialSight Intelligence is a domain-focused RAG application for clinical evidence review. Instead of building a generic chatbot, this project wraps an LLM in a retrieval, security, and cost-control layer so users can query their own document corpus with grounded answers, citations, persistence, and abuse protection.

In short: it shows product thinking, backend architecture, RAG design, multi-user isolation, and deployment awareness in one project.
