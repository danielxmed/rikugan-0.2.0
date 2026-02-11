from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from backend import config
from backend.adapters import AdapterRegistry
from backend.state import server_state

router = APIRouter()


@router.get("/models")
def list_models():
    models = []
    for info in AdapterRegistry.list_all():
        models.append({
            **asdict(info),
            "loaded": server_state.active_adapter_id == info.adapter_id,
        })
    return models


@router.post("/models/{model_name}/load")
def load_model(model_name: str):
    adapter_cls = AdapterRegistry.resolve(model_name)
    if adapter_cls is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    info = adapter_cls.info()

    if server_state.active_adapter_id == info.adapter_id and server_state.active_adapter and server_state.active_adapter.is_loaded():
        return {
            "status": "ok",
            "adapter_id": info.adapter_id,
            "message": f"{info.display_name} already loaded on {config.DEVICE}.",
        }

    adapter = adapter_cls()
    adapter.load(
        device=config.DEVICE,
        dtype_str=config.DTYPE,
        cache_dir=config.MODEL_CACHE_DIR,
        hf_token=config.HF_TOKEN,
    )
    server_state.set_adapter(adapter, info.adapter_id)
    return {
        "status": "ok",
        "adapter_id": info.adapter_id,
        "message": f"{info.display_name} loaded on {config.DEVICE}.",
    }


@router.get("/models/{model_name}/info")
def model_info(model_name: str):
    adapter_cls = AdapterRegistry.resolve(model_name)
    if adapter_cls is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")
    return asdict(adapter_cls.info())
