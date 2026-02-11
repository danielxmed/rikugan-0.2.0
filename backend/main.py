from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import config
from backend.adapters import AdapterRegistry
from backend.routes import inference, models, ws

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Rikugan", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router)
app.include_router(inference.router)
app.include_router(ws.router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.on_event("startup")
def on_startup():
    AdapterRegistry.discover()
    logger.info(
        "Discovered adapters: %s",
        [info.adapter_id for info in AdapterRegistry.list_all()],
    )


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=config.HOST,
        port=config.PORT,
        loop="uvloop",
        ws_max_size=419430400,
        ws_ping_timeout=60,
    )
