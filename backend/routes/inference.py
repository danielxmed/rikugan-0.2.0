from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.state import server_state

router = APIRouter()


class RunRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64


@router.post("/inference/run")
def run_inference(req: RunRequest):
    if server_state.active_adapter is None or not server_state.active_adapter.is_loaded():
        raise HTTPException(status_code=400, detail="No model loaded. Use /models/{name}/load first.")

    generated = server_state.active_adapter.generate(req.prompt, req.max_new_tokens)
    return {
        "prompt": req.prompt,
        "generated_text": generated,
        "model_id": server_state.active_adapter_id,
    }
