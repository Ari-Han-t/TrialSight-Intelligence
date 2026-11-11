from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.auth import router as auth_router
from app.api.routes.rag import router as rag_router
from app.core.config import get_settings
from app.core.database import init_db
from app.rate_limit.middleware import RateLimitMiddleware
from app.services.retrieval import retrieval_service


settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path("./data").mkdir(parents=True, exist_ok=True)
    init_db()
    retrieval_service.warm()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def public_config() -> dict[str, str]:
    """Non-secret runtime info shown in the UI."""
    from app.services.embeddings import get_embedder
    from app.services.llm import llm_service

    return {"llm_provider": llm_service.provider, "embedder": get_embedder().name}


app.include_router(auth_router)
app.include_router(rag_router)

if settings.serve_frontend and Path(settings.frontend_dir).is_dir():
    # Mounted last so API routes always win; html=True serves index.html at "/".
    app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
