from __future__ import annotations

from backend.adapters import ModelAdapter


class ServerState:
    def __init__(self) -> None:
        self.active_adapter: ModelAdapter | None = None
        self.active_adapter_id: str | None = None

    def set_adapter(self, adapter: ModelAdapter, adapter_id: str) -> None:
        if self.active_adapter is not None and self.active_adapter_id != adapter_id:
            self.active_adapter.unload()
        self.active_adapter = adapter
        self.active_adapter_id = adapter_id

    def clear(self) -> None:
        if self.active_adapter is not None:
            self.active_adapter.unload()
        self.active_adapter = None
        self.active_adapter_id = None


server_state = ServerState()
