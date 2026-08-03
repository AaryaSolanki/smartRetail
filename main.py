"""
Module C entrypoint: unified FastAPI gateway.

- /recognize-face      -> app/routers/vision.py
- /classify-product    -> app/routers/vision.py
- /analyze-sentiment   -> app/routers/nlp.py
- /chatbot             -> app/routers/chatbot.py
- /dashboard/stats     -> defined below
- /health              -> liveness check for deployment platforms

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Swagger docs auto-available at /docs, ReDoc at /redoc.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import db
from app.schemas import DashboardStats
from app.routers import vision, nlp, chatbot
from app.services import cv_utils, nlp_utils, chatbot_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # cv_service / nlp_service / chatbot_service already loaded their models
    # at import time (singleton pattern) — this just confirms + initializes storage.
    db.init_db()
    logger.info("Storage initialized.")
    logger.info(f"CV models status: {cv_service.models_status()}")
    logger.info(f"NLP model status: {nlp_service.model_status()}")
    logger.info(f"Chatbot model status: {chatbot_service.model_status()}")
    yield


app = FastAPI(
    title="Smart Retail & Customer Intelligence Platform",
    description="Unified API for face recognition, product classification, "
                "sentiment analysis, and a FAQ chatbot.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: relax for a demo dashboard/frontend; tighten allow_origins in real production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(vision.router)
app.include_router(nlp.router)
app.include_router(chatbot.router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/dashboard/stats", response_model=DashboardStats)
def dashboard_stats():
    stats = db.get_stats()
    models_loaded = {
        **cv_service.models_status(),
        **nlp_service.model_status(),
        **chatbot_service.model_status(),
    }
    return DashboardStats(**stats, models_loaded=models_loaded)
