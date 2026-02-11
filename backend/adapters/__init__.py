from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelInfo:
    adapter_id: str
    display_name: str
    hf_repo: str
    aliases: tuple[str, ...]
    layers: int
    heads: int
    kv_heads: int
    d_model: int
    d_intermediate: int
    vocab_size: int
    max_seq_len: int
    parameters_approx: str


class ModelAdapter(ABC):
    @staticmethod
    @abstractmethod
    def info() -> ModelInfo: ...

    @abstractmethod
    def load(self, device: str, dtype_str: str, cache_dir: str, hf_token: str | None) -> None: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 64) -> str: ...

    @abstractmethod
    def module_path_for_layer(self, layer_idx: int) -> str: ...

    @abstractmethod
    def module_path_for_attn(self, layer_idx: int) -> str: ...

    @abstractmethod
    def module_path_for_mlp(self, layer_idx: int) -> str: ...


class AdapterRegistry:
    _adapters: dict[str, type[ModelAdapter]] = {}   # adapter_id -> class
    _aliases: dict[str, str] = {}                    # alias -> adapter_id

    @classmethod
    def discover(cls) -> None:
        cls._adapters.clear()
        cls._aliases.clear()
        package_dir = Path(__file__).parent
        for module_info in pkgutil.iter_modules([str(package_dir)]):
            if module_info.name.startswith("_"):
                continue
            mod = importlib.import_module(f"backend.adapters.{module_info.name}")
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, ModelAdapter)
                    and attr is not ModelAdapter
                ):
                    info = attr.info()
                    if info.adapter_id in cls._adapters:
                        raise RuntimeError(f"Duplicate adapter_id: {info.adapter_id}")
                    cls._adapters[info.adapter_id] = attr
                    for alias in info.aliases:
                        lower = alias.lower()
                        if lower in cls._aliases:
                            raise RuntimeError(
                                f"Alias collision: '{alias}' claimed by both "
                                f"'{cls._aliases[lower]}' and '{info.adapter_id}'"
                            )
                        cls._aliases[lower] = info.adapter_id

    @classmethod
    def resolve(cls, name: str) -> type[ModelAdapter] | None:
        lower = name.lower()
        if lower in cls._adapters:
            return cls._adapters[lower]
        adapter_id = cls._aliases.get(lower)
        if adapter_id:
            return cls._adapters[adapter_id]
        return None

    @classmethod
    def list_all(cls) -> list[ModelInfo]:
        return [adapter_cls.info() for adapter_cls in cls._adapters.values()]
