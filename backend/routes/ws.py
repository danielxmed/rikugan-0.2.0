from __future__ import annotations

import json
import logging
import struct

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.state import server_state

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.active.remove(ws)

    async def send_json(self, ws: WebSocket, data: dict) -> None:
        await ws.send_text(json.dumps(data))

    async def send_bytes(self, ws: WebSocket, data: bytes) -> None:
        await ws.send_bytes(data)

    async def broadcast_json(self, data: dict) -> None:
        text = json.dumps(data)
        for ws in self.active:
            await ws.send_text(text)


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_json(ws, {
                    "type": "error",
                    "payload": {"message": "Invalid JSON"},
                })
                continue

            msg_type = msg.get("type")
            payload = msg.get("payload", {})

            if msg_type == "inference.run":
                prompt = payload.get("prompt", "")
                max_new_tokens = payload.get("max_new_tokens", 64)

                if server_state.active_adapter is None or not server_state.active_adapter.is_loaded():
                    await manager.send_json(ws, {
                        "type": "error",
                        "payload": {"message": "No model loaded."},
                    })
                    continue

                adapter = server_state.active_adapter
                model_id = server_state.active_adapter_id

                if hasattr(adapter, "forward_with_slices"):
                    generated, block_heat, slice_bytes, slice_meta, token_proj_bytes, dim_proj_bytes = (
                        adapter.forward_with_slices(prompt, max_new_tokens)
                    )

                    # 1. JSON activation.frame (backward compat)
                    await manager.send_json(ws, {
                        "type": "activation.frame",
                        "payload": {
                            "block_heat": block_heat,
                            "model_id": model_id,
                            "prompt": prompt,
                            "num_layers": len(block_heat),
                        },
                    })

                    # 2. JSON activation.slices metadata
                    await manager.send_json(ws, {
                        "type": "activation.slices",
                        "payload": slice_meta,
                    })

                    # 3. Binary frame: 4-byte tag 0x01 + slice data
                    if slice_bytes:
                        tagged_slices = struct.pack('<I', 0x01) + slice_bytes
                        await manager.send_bytes(ws, tagged_slices)

                    # 4. JSON activation.projections metadata
                    if token_proj_bytes and dim_proj_bytes:
                        await manager.send_json(ws, {
                            "type": "activation.projections",
                            "payload": {
                                "num_layers": len(block_heat),
                                "seq_len": slice_meta.get("seq_len", 0),
                                "d_model": slice_meta.get("d_model", 0),
                                "num_stages": 6,
                                "token_proj_size": len(token_proj_bytes),
                            },
                        })

                        # 5. Binary frame: 4-byte tag 0x02 + token_proj + dim_proj
                        tagged_proj = struct.pack('<I', 0x02) + token_proj_bytes + dim_proj_bytes
                        await manager.send_bytes(ws, tagged_proj)

                elif hasattr(adapter, "generate_with_hooks"):
                    generated, block_heat = adapter.generate_with_hooks(prompt, max_new_tokens)
                    await manager.send_json(ws, {
                        "type": "activation.frame",
                        "payload": {
                            "block_heat": block_heat,
                            "model_id": model_id,
                            "prompt": prompt,
                            "num_layers": len(block_heat),
                        },
                    })
                else:
                    generated = adapter.generate(prompt, max_new_tokens)

                # 4. JSON inference.result
                await manager.send_json(ws, {
                    "type": "inference.result",
                    "payload": {
                        "text": generated,
                        "model_id": model_id,
                    },
                })
            else:
                await manager.send_json(ws, {
                    "type": "error",
                    "payload": {"message": f"Unknown message type: {msg_type}"},
                })

    except WebSocketDisconnect:
        manager.disconnect(ws)
